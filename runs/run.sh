#!/bin/bash

LRS=(0.0001)
KLWS=(1)
PARTITIONS=("LV")
SCALED_=("scaled" "non_scaled")
NTRAINING_=(5000)
Z_=(4 8 16)

for i in `seq 1 3`; do
for NTRAINING in ${NTRAINING_[@]}; do
for SCALED in ${SCALED_[@]}; do
for PARTITION in ${PARTITIONS[@]}; do
for LR in ${LRS[@]}; do
for KLW in ${KLWS[@]}; do
for Z in ${Z_[@]}; do
    echo "$PARTITION $SCALED nz=$Z: LR $LR - KLW $KLW - nTraining $NTRAINING"
    python main.py --nTraining $NTRAINING --nVal 1000 --epoch 500 \
    --z ${Z} --learning_rate ${LR} --kld_weight ${KLW} \
    --partition ${PARTITION} --phase ED \
    --preprocessed_data data/transforms/cached/2ch_segmentation__${PARTITION}__ED__${SCALED}.pkl \
    --procrustes_scaling &>> kk.log &
    sleep 15
done
done
done
done
done
done
done
