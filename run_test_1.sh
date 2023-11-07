#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
OUTPUT='/home/won/workspace/graduation/Conformer_xai/ckpt/1103_epoch100/'

python main_test_xai.py --test --model Conformer_tiny_patch16 --batch-size 1 \
    --finetune /home/won/workspace/graduation/Conformer_xai/ckpt/1103_epoch100/checkpoint_epoch006_macroacc53.82_microacc85.59.pth \
	--output_dir ${OUTPUT}

python main_test_xai.py --test --model Conformer_tiny_patch16 --batch-size 1 \
    --finetune /home/won/workspace/graduation/Conformer_xai/ckpt/1103_epoch100/checkpoint_epoch005_macroacc50.46_microacc79.66.pth \
	--output_dir ${OUTPUT}

python main_test_xai.py --test --model Conformer_tiny_patch16 --batch-size 1 \
    --finetune /home/won/workspace/graduation/Conformer_xai/ckpt/1103_epoch100/checkpoint_epoch004_macroacc34.34_microacc76.84.pth \
	--output_dir ${OUTPUT}

python main_test_xai.py --test --model Conformer_tiny_patch16 --batch-size 1 \
    --finetune /home/won/workspace/graduation/Conformer_xai/ckpt/1103_epoch100/checkpoint_epoch000_macroacc33.33_microacc76.27.pth \
	--output_dir ${OUTPUT}

