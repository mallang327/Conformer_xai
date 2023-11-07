"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
import pdb
from sklearn import metrics
import numpy as np
import pandas as pd

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True
                    ):
    # TODO fix this for finetuning
    model.train(set_training_mode)
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True) # [64,3,224,224]
        targets = targets.to(device, non_blocking=True) # [64]

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if isinstance(outputs, list):
                loss_list = [criterion(o, targets) / len(outputs) for o in outputs]
                loss = sum(loss_list)
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if isinstance(outputs, list):
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
        else:
            metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, nb_classes=None):
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1.6,0.6,4.0]).to(device))

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_targets = torch.tensor([])
    all_pred = torch.tensor([])
    all_pred_conv = torch.tensor([])
    all_pred_trans = torch.tensor([])

    _metrics = {}
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True) # torch.Size([64, 1, 224, 224])
        target = target.to(device, non_blocking=True) # torch.Size([64])
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images) # output[0].shape = torch.Size([64, 3])
            # Conformer
            if isinstance(output, list):
                loss_list = [criterion(o, target) / len(output)  for o in output]
                loss = sum(loss_list)
            # others
            else:
                loss = criterion(output, target)
        
        # Conformer
        _, pred_conv = output[0].topk(1, 1, True, True)
        _, pred_trans = output[1].topk(1, 1, True, True)
        _, pred = (output[0]+output[1]).topk(1, 1, True, True)
        pred_conv, pred_trans, pred = pred_conv.t().cpu(), pred_trans.t().cpu(), pred.t().cpu()

        all_targets = torch.cat((all_targets, target.cpu()), dim=0)        
        all_pred_conv = torch.cat((all_pred_conv, pred_conv.squeeze()), dim=0)
        all_pred_trans = torch.cat((all_pred_trans, pred_trans.squeeze()), dim=0)
        all_pred = torch.cat((all_pred, pred.squeeze()), dim=0)


        acc1_head1 = accuracy(output[0], target, topk=(1,))[0]
        acc1_head2 = accuracy(output[1], target, topk=(1,))[0]
        acc1_total = accuracy(output[0] + output[1], target, topk=(1,))[0]


        batch_size = images.shape[0]

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_0=loss_list[0].item())
        metric_logger.update(loss_1=loss_list[1].item())
        metric_logger.meters['acc1'].update(acc1_total.item(), n=batch_size)
        metric_logger.meters['acc1_head1'].update(acc1_head1.item(), n=batch_size)
        metric_logger.meters['acc1_head2'].update(acc1_head2.item(), n=batch_size)


    if nb_classes:
        #maxk = 1
        target_unsqueeze = all_targets.reshape(1,-1).expand_as(all_pred.unsqueeze(0)).cpu()
        #correct = pred.eq(target_unsqueeze)
        #pdb.set_trace()
        for c in range(nb_classes):
            mask = target_unsqueeze == c
            tum = target_unsqueeze[mask]
            # 해당 label에 대한 target이 없는 경우
            #if len(tum) == 0:
            #    _metrics[f"acc1_total_label{c}"] = np.nan
            #    _metrics[f"acc1_head1_label{c}"] = np.nan
            #    _metrics[f"acc1_head2_label{c}"] = np.nan
            #    continue

            _metrics[f"acc1_total_label{c}"] = metrics.accuracy_score(tum, all_pred.unsqueeze(0)[mask]) * 100
            _metrics[f"acc1_head1_label{c}"] = metrics.accuracy_score(tum, all_pred_conv.unsqueeze(0)[mask]) * 100
            _metrics[f"acc1_head2_label{c}"] = metrics.accuracy_score(tum, all_pred_trans.unsqueeze(0)[mask]) * 100
        
        try:
            _metrics[f"recall_macro_total"] = metrics.recall_score(all_targets, all_pred, average='macro') * 100
            _metrics[f"recall_macro_head1"] = metrics.recall_score(all_targets, all_pred_conv, average='macro') * 100
            _metrics[f"recall_macro_head2"] = metrics.recall_score(all_targets, all_pred_trans, average='macro') * 100
        except:
            pdb.set_trace()
        # label 1개가 비어있는 경우 고려해야함
        # for c in range(nb_classes):
        #     _metrics[f"recall_macro_total"] += _metrics[f"acc1_total_label{c}"]
        #     _metrics[f"recall_macro_head1"] += _metrics[f"acc1_head1_label{c}"]
        #     _metrics[f"recall_macro_head2"] += _metrics[f"acc1_head2_label{c}"]
        for k, v in _metrics.items():
            if v == np.nan:
                continue
            metric_logger.meters[k].update(v, n=batch_size)

    if isinstance(output, list):
        print('* Acc@heads_top1 {heads_top1.global_avg:.3f} Acc@head_1 {head1_top1.global_avg:.3f} Acc@head_2 {head2_top1.global_avg:.3f} '
              'loss@total {losses.global_avg:.3f} loss@1 {loss_0.global_avg:.3f} loss@2 {loss_1.global_avg:.3f} '
              'Acc_0@total {acc1_total_0.global_avg:.3f} Acc_1@total {acc1_total_1.global_avg:.3f} Acc_2@total {acc1_total_2.global_avg:.3f} macro_recall@total {macro_recall_total.global_avg:.3f}'
              .format(heads_top1=metric_logger.acc1, head1_top1=metric_logger.acc1_head1, head2_top1=metric_logger.acc1_head2,
                      losses=metric_logger.loss, loss_0=metric_logger.loss_0, loss_1=metric_logger.loss_1,
                      acc1_total_0=metric_logger.acc1_total_label0, acc1_total_1=metric_logger.acc1_total_label1, acc1_total_2=metric_logger.acc1_total_label2,
                      macro_recall_total=metric_logger.recall_macro_total))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test(data_loader, model, device, nb_classes=None, testset=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_targets = torch.tensor([])
    all_pred = torch.tensor([])
    all_pred_conv = torch.tensor([])
    all_pred_trans = torch.tensor([])
    all_files = []

    _metrics = {}
    for images, target, files in metric_logger.log_every(data_loader, 100, header):
        all_files = all_files + files
        images = images.to(device, non_blocking=True) # torch.Size([64, 1, 224, 224])
        target = target.to(device, non_blocking=True) # torch.Size([64])
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images) # output[0].shape = torch.Size([64, 3])
            # Conformer
            if isinstance(output, list):
                loss_list = [criterion(o, target) / len(output)  for o in output]
                loss = sum(loss_list)
            # others
            else:
                loss = criterion(output, target)
        
        # Conformer
        _, pred_conv = output[0].topk(1, 1, True, True)
        _, pred_trans = output[1].topk(1, 1, True, True)
        _, pred = (output[0]+output[1]).topk(1, 1, True, True)
        pred_conv, pred_trans, pred = pred_conv.t().cpu(), pred_trans.t().cpu(), pred.t().cpu()

        all_targets = torch.cat((all_targets, target.cpu()), dim=0)        
        all_pred_conv = torch.cat((all_pred_conv, pred_conv.squeeze(0)), dim=0)
        all_pred_trans = torch.cat((all_pred_trans, pred_trans.squeeze(0)), dim=0)
        all_pred = torch.cat((all_pred, pred.squeeze(0)), dim=0)


        acc1_head1 = accuracy(output[0], target, topk=(1,))[0]
        acc1_head2 = accuracy(output[1], target, topk=(1,))[0]
        acc1_total = accuracy(output[0] + output[1], target, topk=(1,))[0]


        batch_size = images.shape[0]

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_0=loss_list[0].item())
        metric_logger.update(loss_1=loss_list[1].item())
        metric_logger.meters['acc1'].update(acc1_total.item(), n=batch_size)
        metric_logger.meters['acc1_head1'].update(acc1_head1.item(), n=batch_size)
        metric_logger.meters['acc1_head2'].update(acc1_head2.item(), n=batch_size)


    if nb_classes:
        #maxk = 1
        target_unsqueeze = all_targets.reshape(1,-1).expand_as(all_pred.unsqueeze(0)).cpu()
        #correct = pred.eq(target_unsqueeze)
        #pdb.set_trace()
        for c in range(nb_classes):
            mask = target_unsqueeze == c
            tum = target_unsqueeze[mask]
            # 해당 label에 대한 target이 없는 경우
            #if len(tum) == 0:
            #    _metrics[f"acc1_total_label{c}"] = np.nan
            #    _metrics[f"acc1_head1_label{c}"] = np.nan
            #    _metrics[f"acc1_head2_label{c}"] = np.nan
            #    continue

            _metrics[f"acc1_total_label{c}"] = metrics.accuracy_score(tum, all_pred.unsqueeze(0)[mask]) * 100
            _metrics[f"acc1_head1_label{c}"] = metrics.accuracy_score(tum, all_pred_conv.unsqueeze(0)[mask]) * 100
            _metrics[f"acc1_head2_label{c}"] = metrics.accuracy_score(tum, all_pred_trans.unsqueeze(0)[mask]) * 100
        
        try:
            _metrics[f"recall_macro_total"] = metrics.recall_score(all_targets, all_pred, average='macro') * 100
            _metrics[f"recall_macro_head1"] = metrics.recall_score(all_targets, all_pred_conv, average='macro') * 100
            _metrics[f"recall_macro_head2"] = metrics.recall_score(all_targets, all_pred_trans, average='macro') * 100
        except:
            pdb.set_trace()
        # label 1개가 비어있는 경우 고려해야함
        # for c in range(nb_classes):
        #     _metrics[f"recall_macro_total"] += _metrics[f"acc1_total_label{c}"]
        #     _metrics[f"recall_macro_head1"] += _metrics[f"acc1_head1_label{c}"]
        #     _metrics[f"recall_macro_head2"] += _metrics[f"acc1_head2_label{c}"]
        for k, v in _metrics.items():
            if v == np.nan:
                continue
            metric_logger.meters[k].update(v, n=batch_size)

    if isinstance(output, list):
        print('* Acc@heads_top1 {heads_top1.global_avg:.3f} Acc@head_1 {head1_top1.global_avg:.3f} Acc@head_2 {head2_top1.global_avg:.3f} '
              'loss@total {losses.global_avg:.3f} loss@1 {loss_0.global_avg:.3f} loss@2 {loss_1.global_avg:.3f} '
              'Acc_0@total {acc1_total_0.global_avg:.3f} Acc_1@total {acc1_total_1.global_avg:.3f} Acc_2@total {acc1_total_2.global_avg:.3f} macro_recall@total {macro_recall_total.global_avg:.3f}'
              .format(heads_top1=metric_logger.acc1, head1_top1=metric_logger.acc1_head1, head2_top1=metric_logger.acc1_head2,
                      losses=metric_logger.loss, loss_0=metric_logger.loss_0, loss_1=metric_logger.loss_1,
                      acc1_total_0=metric_logger.acc1_total_label0, acc1_total_1=metric_logger.acc1_total_label1, acc1_total_2=metric_logger.acc1_total_label2,
                      macro_recall_total=metric_logger.recall_macro_total))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    predlabel_dict = {"file": all_files, "pred": all_pred, "pred_conv": all_pred_conv,
                 "pred_trans": all_pred_trans, "label": all_targets}
    predlabel = pd.DataFrame(predlabel_dict)
    if testset == "test":
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, predlabel
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
