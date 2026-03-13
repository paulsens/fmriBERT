#!/bin/bash
# Launcher for paired pretraining jobs.
# Usage: bash scripts/paired_pretrain.sh "description" [--folds 0-7] [--lr 0.00001] [--task CLS_only] [--dataset audimg]

cd /path/to/fmriBERT/scripts || exit

MSG="${1:?Please provide a description as the first argument}"
FOLDS="${2:-0-7}"
LR="${3:-0.00001}"
TASK="${4:-CLS_only}"
DATASET="${5:-audimg}"
ATN_HEADS="${6:-2}"
NUM_LAYERS="${7:-3}"
FWD_EXP="${8:-4}"
SEQ_LEN="${9:-5}"
SAVE="${10:-True}"
CLS_WEIGHT="${11:-1}"

IFS='-' read -r FOLD_START FOLD_END <<< "$FOLDS"
FOLD_END="${FOLD_END:-$FOLD_START}"

for i in $(seq "$FOLD_START" "$FOLD_END"); do
    sbatch paired_pretrain.script "$MSG" "$i" "$CLS_WEIGHT" "$LR" "$SAVE" "$TASK" "$ATN_HEADS" "$NUM_LAYERS" "$FWD_EXP" "$SEQ_LEN" "$DATASET"
    sleep 2
done
