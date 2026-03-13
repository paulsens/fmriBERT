#!/bin/bash
# Unified launcher for finetuning jobs (sametimbre, samegenre, etc.)
# Usage: bash scripts/finetune.sh "description" <task> [--pretrain_task both] [--pti 0] [--folds 0-4] [--freeze False]
#
# Examples:
#   bash scripts/finetune.sh "exp1" sametimbre --pretrain_task both --pti 8 --folds 0-4
#   bash scripts/finetune.sh "exp1" samegenre --pretrain_task both --pti 10 --folds 0-4
#   bash scripts/finetune.sh "exp1" sametimbre --pretrain_task fresh --pti fresh --folds 0

cd /path/to/fmriBERT/scripts || exit

MSG="${1:?Please provide a description as the first argument}"
TASK="${2:?Please provide a finetune task (sametimbre, samegenre)}"
PRETRAIN_TASK="${3:-both}"
PTI="${4:-0}"
FOLDS="${5:-0-4}"
FREEZE="${6:-False}"

# Map task name to python script
case "$TASK" in
    sametimbre) SCRIPT="sametimbre.py" ;;
    samegenre)  SCRIPT="samegenre.py" ;;
    *)          echo "Unknown task: $TASK"; exit 1 ;;
esac

IFS='-' read -r FOLD_START FOLD_END <<< "$FOLDS"
FOLD_END="${FOLD_END:-$FOLD_START}"

for i in $(seq "$FOLD_START" "$FOLD_END"); do
    sbatch finetune.script "$MSG" "$PRETRAIN_TASK" "$PTI" "$i" "$FREEZE" "$SCRIPT"
    sleep 4
done
