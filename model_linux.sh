#!/bin/bash

# Define variables
MODEL_ID="MUSIC-2mix-LogFreq-resnet18dilated-unet7-linear-frames3stride24-maxpool-binary-weightedLoss-channels32-epoch100-step40_80"
MODEL_PATH="./ckpt/$MODEL_ID"
LINK_RELEASE="http://sound-of-pixels.csail.mit.edu/release"

LIST_VAL="$LINK_RELEASE/val.csv"
WEIGHTS_FRAME="$LINK_RELEASE/${MODEL_ID}/frame_best.pth"
WEIGHTS_SOUND="$LINK_RELEASE/${MODEL_ID}/sound_best.pth"
WEIGHTS_SYNTHESIZER="$LINK_RELEASE/${MODEL_ID}/synthesizer_best.pth"

# Create necessary directories
mkdir -p ./data
mkdir -p "$MODEL_PATH"

# Download validation list and model weights
wget -O ./data/val.csv "$LIST_VAL"
wget -O "$MODEL_PATH/sound_best.pth" "$WEIGHTS_SOUND"
wget -O "$MODEL_PATH/frame_best.pth" "$WEIGHTS_FRAME"
wget -O "$MODEL_PATH/synthesizer_best.pth" "$WEIGHTS_SYNTHESIZER"

echo "Download complete!"
