## Notice
The code is forked from official [project](https://github.com/open-mmlab/mmdetection). So the basic install and usage of mmdetection can be found in [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md). We only add Conformer as a backbone in `mmdet/models/backbones/Conformer.py`.

At present, we use the feature maps of different stages in the CNN branch as the input of FPN, so that it can be quickly applied to the detection algorithm based on the feature pyramid. At the same time, we think that how to use the features of Transformer branch for detection is also an interesting problem.

## Training and inference under different detction algorithms
We provide some config files in `configs`. And anyone can use Conformer to replace the backbone in the existing detection algorithms. Below we will take the `Faster R-CNN` algorithm as an example to illustrate how to perform training and inference:

```bash
export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7
export OMP_NUM_THREADS=1
GPU_NUM=8

CONFIG="./configs/faster_rcnn/faster_rcnn_conformer_small_patch32_fpn_1x_coco.py"

WORK_DIR='./work_dir/faster_rcnn_conformer_small_patch32_lr_1e_4_fpn_1x_coco_1344_800'


python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50040 --use_env ./tools/train.py ${CONFIG} --work-dir ${WORK_DIR} --gpus ${GPU_NUM}  --launcher pytorch --cfg-options model.pretrained='./pretrain_models/Conformer_small_patch32.pth' model.backbone.patch_size=32


python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50040 --use_env ./tools/test.py ${CONFIG} ${WORK_DIR}/latest.pth --launcher pytorch  --eval bbox

```

Here, we use the Conformer_small_patch32 as backbone, whose pretrain model weight can be downloaded from [baidu]() or [google]().
.