#!/bin/bash

#Move to project directory
cd /isi/music/auditoryimagery2/seanthesis/timedir/scripts || exit

#submit to the queue
for fold in {0..0}
do
  for LR in 0.0000001
  do
    for atn_heads in 4
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
       sbatch unpaireddecoding.script "$1" "$fold" "False" $LR "False" "fresh" $atn_heads "2" "4" "fresh"
       sleep 2
    done
  done
done

