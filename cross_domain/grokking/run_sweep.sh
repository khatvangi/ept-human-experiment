#!/bin/bash
# run grokking sweep on boron GPU
# 10 seeds × 6 weight decay values = 60 runs
# estimated time: ~1 hour on Titan RTX

cd /storage/EPT/ept_human_experiment/cross_domain/grokking/

# make output directory for this sweep
mkdir -p runs

echo "starting grokking sweep at $(date)"
python3 train_grokking.py \
    --seeds 10 \
    --gpu 0 \
    --epochs 7500 \
    --output_dir runs/ \
    2>&1 | tee runs/sweep_log.txt

echo "finished at $(date)"
