#!/bin/bash
# Launcher for OpenGenre pretraining jobs.
# Usage: bash scripts/pretrain.sh "description" [--folds 0-11] [--binweight 0.5] [--lr 0.0001]

cd /path/to/fmriBERT/scripts || exit

MSG="${1:?Please provide a description as the first argument}"
FOLDS="${2:-0}"        # e.g. "0-11" or "0" (default: single fold 0)
BINWEIGHT="${3:-0.5}"
LR="${4:-0.0001}"

# Parse fold range
IFS='-' read -r FOLD_START FOLD_END <<< "$FOLDS"
FOLD_END="${FOLD_END:-$FOLD_START}"

for i in $(seq "$FOLD_START" "$FOLD_END"); do
    sbatch pretrain.script "$MSG" "$i" "$BINWEIGHT" "$LR"
    sleep 2
done
