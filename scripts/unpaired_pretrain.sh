#!/bin/bash
# Launcher for unpaired pretraining jobs.
# Usage: bash scripts/unpaired_pretrain.sh "description" [--folds 0-7] [--lr 0.00001] [--task both] [--dataset audimg]

cd /path/to/fmriBERT/scripts || exit

MSG="${1:?Please provide a description as the first argument}"
FOLDS="${2:-0}"
LR="${3:-0.00001}"
TASK="${4:-both}"
DATASET="${5:-audimg}"
ATN_HEADS="${6:-4}"
NUM_LAYERS="${7:-3}"
FWD_EXP="${8:-4}"
SAVE="${9:-False}"
CLS_WEIGHT="${10:-1}"

IFS='-' read -r FOLD_START FOLD_END <<< "$FOLDS"
FOLD_END="${FOLD_END:-$FOLD_START}"

for fold in $(seq "$FOLD_START" "$FOLD_END"); do
    sbatch unpaired_pretrain.script "$MSG" "$fold" "$CLS_WEIGHT" "$LR" "$SAVE" "$TASK" "$ATN_HEADS" "$NUM_LAYERS" "$FWD_EXP" "$DATASET"
    sleep 2
done
