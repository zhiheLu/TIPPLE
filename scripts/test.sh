#!/bin/bash

# 'I', 'A', 'K', 'R', 'V', 'DTD', 'Flower102', 'Food101', 'Cars', 'SUN397', 'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat'

data_root='/mnt/c/Users/Lucas/Research/Datasets'

output_root=output

arch=RN50 # ViT-B/16
s1_bs=256
ctx_init=a_photo_of_a
s1_lr=0.001
s1_epoch=3
s1_thred=0.7
s1_text_loss=0.0
s1_div_loss=0.2 # 0.1 for OOD datasets and 0.2 for Non-OOD datasets
s2_lr=0.001
s2_views=64
s2_selection_p=0.1
gpu_id=0

for SEED in 3 # 1 2 3
do
for DATASET in Caltech101 # DTD Pets Cars UCF101 Flower102 Food101 SUN397 Aircraft EuroSAT I A K R V
do
CUDA_VISIBLE_DEVICES=${gpu_id} python -u ./main.py \
${data_root} \
--dataset ${DATASET} \
--arch ${arch} \
--s1-batch-size ${s1_bs} \
--s1-learning-rate ${s1_lr} \
--s1-epoch ${s1_epoch} \
--s1-thred ${s1_thred} \
--s1-text-loss-weight ${s1_text_loss} \
--s1-div-loss-weight ${s1_div_loss} \
--s2-views ${s2_views} \
--s2-learning-rate ${s2_lr} \
--s2-selection-p ${s2_selection_p} \
--output_dir ${output_root}/${DATASET}/${arch}/s1epoch${s1_epoch}_s1lr${s1_lr}_s1bs${s1_bs}_dynamicdiv${s1_div_loss}_text${s1_text_loss}_thred${s1_thred}_s2lr${s2_lr}/seed${SEED} \
--ctx_init ${ctx_init} \
--seed ${SEED} \
--print-freq 5
done
done
