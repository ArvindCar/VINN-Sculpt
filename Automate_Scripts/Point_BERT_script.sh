#!/bin/bash

# Set the path to the Python script
script_path="/home/arvind/CMU/MAIL/VINN/VINN-Main/VINN-Sculpt/VINN-ACT/representation_models/BYOL_PointBERT.py"

# Set other common parameters
root_dir="/home/arvind/CMU/MAIL/VINN/VINN-Main/VINN-Sculpt/VINN-ACT/"
folder="/home/arvind/CMU/MAIL/VINN/VINN-Main/VINN-Sculpt/X_Datasets"
dataset="X_Datasets"
extension="X"
img_size=2048
epochs=101
wandb=1
gpu=1
hidden_layer=-1
pretrained=1
save_dir="/home/arvind/CMU/MAIL/VINN/VINN-Main/VINN-Sculpt/X_chkpts/"

# Iterate over batch sizes from 10 to 60 in steps of 10
for batch_size in {20..60..10}
do
    # Execute the command with the current batch size
    python $script_path \
        --batch_size $batch_size \
        --root_dir $root_dir \
        --folder $folder \
        --dataset $dataset \
        --extension $extension \
        --img_size $img_size \
        --epochs $epochs \
        --wandb $wandb \
        --gpu $gpu \
        --hidden_layer $hidden_layer \
        --pretrained $pretrained \
        --save_dir "${save_dir}${batch_size}/"
done
