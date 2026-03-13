#!/bin/bash
# Unified launcher for decoding jobs (timbre_decoding, unpaireddecode, timedir_pitchclass).
# Usage: bash scripts/decode.sh "description" <task> [--folds 0-7] [--lr 0.00001] [--pretrain_task fresh] [--pti fresh]
#
# Examples:
#   bash scripts/decode.sh "exp1" timbre --pretrain_task fresh --pti fresh --lr 0.0000001
#   bash scripts/decode.sh "exp1" unpaired --pretrain_task CLS_only --pti 0

cd /path/to/fmriBERT/scripts || exit

MSG="${1:?Please provide a description as the first argument}"
TASK="${2:?Please provide a decode task (timbre, unpaired, pitchclass)}"
FOLDS="${3:-0}"
LR="${4:-0.00001}"
PRETRAIN_TASK="${5:-fresh}"
PTI="${6:-fresh}"
ATN_HEADS="${7:-4}"
NUM_LAYERS="${8:-3}"
FWD_EXP="${9:-4}"
FREEZE="${10:-False}"
SAVE="${11:-False}"

# Map task name to python script
case "$TASK" in
    timbre)     SCRIPT="timbre_decoding.py" ;;
    unpaired)   SCRIPT="unpaireddecode.py" ;;
    pitchclass) SCRIPT="timedir_pitchclass.py" ;;
    *)          echo "Unknown task: $TASK"; exit 1 ;;
esac

IFS='-' read -r FOLD_START FOLD_END <<< "$FOLDS"
FOLD_END="${FOLD_END:-$FOLD_START}"

for fold in $(seq "$FOLD_START" "$FOLD_END"); do
    sbatch decode.script "$MSG" "$fold" "$FREEZE" "$LR" "$SAVE" "$PRETRAIN_TASK" "$ATN_HEADS" "$NUM_LAYERS" "$FWD_EXP" "$PTI" "$SCRIPT"
    sleep 2
done
