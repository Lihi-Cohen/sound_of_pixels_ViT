#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "
OPTS+="--mode eval "
OPTS+="--list_val data/val_vis.csv "

# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame vit "
OPTS+="--img_pool '' "
OPTS+="--num_channels 32 "
# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 1 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# learning params
OPTS+="--vis_train_mode stage1 "
OPTS+="--num_gpus 1 "
OPTS+="--workers 8 "
OPTS+="--batch_size_per_gpu 1 "


# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

python -u main_eval.py $OPTS
