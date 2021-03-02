#!/bin/bash

cd $(git rev-parse --show-toplevel)

BTCH="100"
LR=0.001
SEEDS=(1 2 3 4 5)
SEEDS=(1)

WSNP=$1
KLW=$2

for SEED in ${SEEDS[@]}; do
  echo $SEED
  python main.py -c output/2020-11-16_13-22-45/config.json \
  --checkpoint_file output/2020-11-16_13-22-45/checkpoints/checkpoint_11.pt \
  --output_dir output/2020-11-16_13-22-45__{TIMESTAMP}__wsnp_${WSNP}__kld_${KLW} \
  --nTraining 5000 --nVal 1000 --batch_size $BTCH --epoch 12 \
  --learning_rate $LR --kld_weight $KLW --seed $RANDOM --snp_weight $WSNP  # &> /dev/null &
  sleep 2
done
