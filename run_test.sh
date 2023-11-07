#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
OUTPUT='/home/won/workspace/graduation/Conformer_xai/ckpt/1103_epoch100/'

python main_test_xai.py --test --model Conformer_tiny_patch16 --batch-size 1 \
    --finetune /home/won/workspace/graduation/Conformer_xai/ckpt/1103_epoch100/checkpoint_epoch059_macroacc65.8_microacc92.94.pth \
	--output_dir ${OUTPUT}

python main_test_xai.py --test --model Conformer_tiny_patch16 --batch-size 1 \
    --finetune /home/won/workspace/graduation/Conformer_xai/ckpt/1103_epoch100/checkpoint_epoch045_macroacc65.16_microacc93.22.pth \
	--output_dir ${OUTPUT}

python main_test_xai.py --test --model Conformer_tiny_patch16 --batch-size 1 \
    --finetune /home/won/workspace/graduation/Conformer_xai/ckpt/1103_epoch100/checkpoint_epoch036_macroacc64.57_microacc90.11.pth \
	--output_dir ${OUTPUT}

python main_test_xai.py --test --model Conformer_tiny_patch16 --batch-size 1 \
    --finetune /home/won/workspace/graduation/Conformer_xai/ckpt/1103_epoch100/checkpoint_epoch031_macroacc63.95_microacc88.7.pth \
	--output_dir ${OUTPUT}

python main_test_xai.py --test --model Conformer_tiny_patch16 --batch-size 1 \
    --finetune /home/won/workspace/graduation/Conformer_xai/ckpt/1103_epoch100/checkpoint_epoch008_macroacc62.28_microacc30.79.pth \
	--output_dir ${OUTPUT}
