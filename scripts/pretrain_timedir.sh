#!/bin/bash
# Launcher for TimeDIR pretraining jobs.
# Usage: bash scripts/pretrain_timedir.sh "description" [--folds 0-7] [--lr 0.00001] [--task CLS_only]

cd /path/to/fmriBERT/scripts || exit

MSG="${1:?Please provide a description as the first argument}"
FOLDS="${2:-0}"
LR="${3:-0.00001}"
TASK="${4:-CLS_only}"
ATN_HEADS="${5:-4}"
NUM_LAYERS="${6:-3}"
FWD_EXP="${7:-4}"
SAVE="${8:-False}"
CLS_WEIGHT="${9:-1}"

IFS='-' read -r FOLD_START FOLD_END <<< "$FOLDS"
FOLD_END="${FOLD_END:-$FOLD_START}"

for heldout_run in $(seq "$FOLD_START" "$FOLD_END"); do
    sbatch pretrain_timedir.script "$MSG" "$heldout_run" "$CLS_WEIGHT" "$LR" "$SAVE" "$TASK" "$ATN_HEADS" "$NUM_LAYERS" "$FWD_EXP"
    sleep 2
done
