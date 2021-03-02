#!/bin/bash

BATCH=("1667" "100" "32")
KLWS=(0.1 0.3 0.5)
LRS=(0.001 0.0005 0.0001)
SEEDS=(1 2 3)
WSNPS=(0.1)

for SEED in ${SEEDS[@]}; do
for BTCH in ${BATCH[@]}; do
for KLW in ${KLWS[@]}; do
for LR in ${LRS[@]}; do
for WSNP in ${WSNPS[@]}; do
python main.py -c output/2020-11-16_13-22-45/config.json \
--checkpoint_file output/2020-11-16_13-22-45/checkpoints/checkpoint_11.pt \
--output_dir output/2020-11-16_13-22-45__{TIMESTAMP} \
--nTraining 5000 --nVal 1000 --batch_size $BTCH --epoch 30 \
--learning_rate $LR --kld_weight $KLW --seed $RANDOM --WSNP $WSNP
done
done
done
done
done
