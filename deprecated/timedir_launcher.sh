#!/bin/bash

#Name of the job
#SBATCH --job-name=pretraintimedir1

#Number of compute nodes
#SBATCH --nodes=1

#Number of tasks per node
#SBATCH --cpus-per-task=2

#Request memory
#SBATCH --mem=20G

#Walltime
#SBATCH --time=24:00:00

#Import environment variables from caller's environment
#SBATCH --export=ALL

#error and output files
#SBATCH -o /isi/music/auditoryimagery2/seanthesis/timedir/pretrain/ofiles/CLS_only/o_%j.txt
#SBATCH -e /isi/music/auditoryimagery2/seanthesis/timedir/pretrain/errfiles/CLS_only/o_%j.err
#SBATCH --mail-user=paulsen.sean@gmail.com
source /Users/sean/miniconda3/envs/fmriprep_venv/bin/activate fmriprep_venv

echo "Description: $1"
echo "Heldout Run: $2"
echo "CLS Task Weight: $3"
echo "Learning Rate: $4"
echo "Save Model: $5"
echo "Task: $6"
echo "Attention Heads: $7"
echo "Num Layers: $8"
echo "Forward Expansion: $9"

python /Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT/pretrain_timedir.py -m "$1" -heldout_run "$2" -CLS_task_weight "$3" -LR "$4" -save_model "$5" -task "$6" -attention_heads "$7" -num_layers "$8" -forward_expansion "$9"
