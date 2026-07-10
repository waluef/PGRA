#!/usr/bin/env bash
set -euo pipefail
PCT="${1:-10}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

SAVE_PATH="runs/densenet_pgra_${PCT}/"

python tools/train.py \
    --train_list data/Train/list/train_${PCT}.txt \
    --val_list data/Train/list/val_${PCT}.txt \
    --ofa1_list data/OFA1_2/list/OFA1.txt \
    --ofa2_list data/OFA1_2/list/OFA2.txt \
    --ofa3_list data/OFA3/list/OFA3.txt \
    --attention_setting \
    --num_runs 5 \
    --num_epochs 250 \
    --patience 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --num_classes 10 \
    --angle_count 5 \
    --angle_interval 2.5 \
    --save_path "${SAVE_PATH}"

echo
echo "==================== Final summary ===================="
python tools/evaluate.py --path "${SAVE_PATH}" | tee "${SAVE_PATH}/summary.txt"
echo "Summary written to ${SAVE_PATH}/summary.txt"
