#!/bin/bash

# called from command line with bash thesis_pipe.sh paired/unpaired pretrain/finetune hold_runs/hold_rand n_folds message_in_quotes
cd /isi/music/auditoryimagery2/seanthesis/scripts/
export pairing=$1
export stage=$2
export holdout=$3
export n_folds=$4
export description=$5

if [$pairing=='paired'] && [$stage=='pretrain']
then


