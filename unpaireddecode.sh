#!/bin/bash

#Move to project directory
cd /isi/music/auditoryimagery2/seanthesis/thesis/scripts || exit

#submit to the queue
for fold in {0..0}
do
  for LR in 0.00001
  do
    for atn_heads in 2
    do
      # 1) $1 is the text message supplied by the user for bookkeeping
      # other arguments, in order, are
      # 2) this count in this script's loop (tells the pythons script which run to hold out as validation),
      # 3) whether to freeze the pretrained model's weights or update them during training
      # 4) the Learning Rate ("default" can be given),
      # 5) whether to save the model at the end of training ("True" or "False" with quotes),
      # 6) pretraining task ("both", "CLS_only", "MSK_only", "fresh")
      # 7) number of attention heads in the model
      # 8) number of layers in the model
      # 9) factor of forward expansion in the model
      # 10) index of pretrained model aka heldout run, can also be "fresh"
      # 11) sequence length without CLS token
      # 12) dataset used in pretraining, usually the same as is used in this phase
      # 13) pretraining task, either both CLS_only MSK_only or fresh
      # 14) decode dataset, should be obvious from decode task, but whatever
      # 15) decode task, e.g genredecode, timbredecode

       sbatch unpaireddecode.script "$1" "$fold" "False" $LR "False" "fresh" $atn_heads "3" "4" "fresh" "10" "genre" "CLS_only" "genre" "genredecode"
       sleep 2
    done
  done
done

