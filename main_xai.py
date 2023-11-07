import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from datasets import build_dataset
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from pathlib import Path

from engine import train_one_epoch, evaluate, test
import time

import utils
import models
import pdb
import json
import pickle

import wandb
import os


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--output_dir', default='/home/won/workspace/graduation/Conformer_xai/ckpt/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--is_wandb', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--wandb_title', default='')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--evaluate-freq', type=int, default=1, help='frequency of perform evaluation (default: 5)')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--apply_padding', action='store_true')
    
    # Model parameters
    parser.add_argument('--model', default='Conformer_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Dataset parameters
    parser.add_argument('--data-path', default='/home/hdd1/won_hdd/DB/autumn_1014/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='autumn', choices=['CIFAR', 'CIFAR10', 'IMNET', 
                                                                  'INAT', 'INAT19', 'autumn', 'autumn_bg'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    
    # * Random Erase params (Datasets)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    return parser
    
def main(args):
    print(args)

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Dataset
    print(f"Loading the dataset...")
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    ## dataset random sample
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    ## dataloader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(args.batch_size),
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
        #batch_size=int(3.0 * args.batch_size)
    )

    ## mixup, cutmix False
    mixup_fn = None

    # Model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
    )
    ## Finetune False
    model.to(device)
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
    model_without_ddp = model
    ## Distributed False

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    ## Optimizer
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = torch.nn.CrossEntropyLoss()
    ## Smoothing False

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(exist_ok=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint.keys():
            model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            model_without_ddp.load_state_dict(checkpoint)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    
    # Training
    print("Start training")
    start_time = time.time()
    max_accuracy = 0.0   

    if args.is_wandb:
        wandb.init(project="Conformer_XAI classification (Autumn)")
        if args.wandb_title:
            wandb.run.name = args.wandb_title
        else:
            wandb.run.name = "[Check] Train tinytiny wandb"
        wandb.run.save()         
        wandb_args = {
            "learning rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs
        }
        wandb.config.update(wandb_args)

    ####### For Safety #######
    pdb.set_trace()

    if not args.no_save_ckpt:
        if args.output_dir and utils.is_main_process():
            pickle.dump(args, open(output_dir / "arguments.pkl", "wb"))

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )
        if args.is_wandb:
            wandb.log({"train_loss_0": train_stats["loss_0"],
                    "train_loss_1": train_stats["loss_1"],
                    "train_lr": train_stats["lr"],
                    })

        lr_scheduler.step(epoch)
        # if args.output_dir:
        #     checkpoint_paths = [output_dir / 'checkpoint.pth']
        #     for checkpoint_path in checkpoint_paths:
        #         utils.save_on_master({
        #             'model': model_without_ddp.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             'model_ema': get_state_dict(model_ema),
        #             'args': args,
        #         }, checkpoint_path)

        if epoch % args.evaluate_freq == 0:
            is_better = False
            test_stats = evaluate(data_loader_val, model, device, args.nb_classes)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            #if max_accuracy >= test_stats["acc1"]:
            #    is_better = True
            #max_accuracy = max(max_accuracy, test_stats["acc1"])
            #print(f'Max accuracy: {max_accuracy:.2f}%')
            if test_stats["recall_macro_total"] > max_accuracy:
               is_better = True
            max_accuracy = max(max_accuracy, test_stats["recall_macro_total"])
            print(f'Max macro accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            if args.is_wandb:
                wandb.log({"val_loss": test_stats["loss"],
                        "val_loss_0": test_stats["loss_0"],
                        "val_loss_1": test_stats["loss_1"],

                        "val_macro_acc_conv": test_stats["recall_macro_head1"],
                        "val_macro_acc_trans": test_stats["recall_macro_head2"],
                        "val_macro_acc_total": test_stats["recall_macro_total"],

                        "val_acc_conv_label0": test_stats["acc1_head1_label0"],
                        "val_acc_conv_label1": test_stats["acc1_head1_label1"],
                        "val_acc_conv_label2": test_stats["acc1_head1_label2"],
                        "val_acc_trans_label1": test_stats["acc1_head2_label1"],
                        "val_acc_trans_label2": test_stats["acc1_head2_label2"],
                        "val_acc_trans_label0": test_stats["acc1_head2_label0"],
                        "val_acc_total_label0": test_stats["acc1_total_label0"],
                        "val_acc_total_label1": test_stats["acc1_total_label1"],
                        "val_acc_total_label2": test_stats["acc1_total_label2"],
                        })

            if not args.no_save_ckpt:
                if (is_better) or (epoch == args.start_epoch):
                    print("Better macro accuracy, Saving the ckpt")
                    test_macro_acc1_round = np.round(test_stats["recall_macro_total"], 2)
                    test_micro_acc1_round = np.round(test_stats["acc1"], 2)
                    checkpoint_paths = [output_dir / f'checkpoint_epoch{str(epoch).zfill(3)}_macroacc{test_macro_acc1_round}_microacc{test_micro_acc1_round}.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'model_ema': get_state_dict(model_ema),
                            'args': args,
                        }, checkpoint_path)

    # Test
    args.finetune = checkpoint_paths[0]
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        if 'model' in checkpoint.keys():
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        #state_dict = model.state_dict()
        model.load_state_dict(checkpoint_model, strict=False)

    args.test = True
    dataset_test, args.nb_classes = build_dataset(is_train=False, args=args)

    ## dataloader
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
        #batch_size=int(3.0 * args.batch_size)
    )
    test_stats, predlabel = test(data_loader_test, model, device, testset="test", nb_classes=args.nb_classes)
    
    print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
    save_csvpath = os.path.join(args.output_dir, str(Path(args.finetune).stem)+'_test.csv')
    predlabel.to_csv(f"{save_csvpath}", index=False)
    print(f"Testset info is saved at {save_csvpath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
