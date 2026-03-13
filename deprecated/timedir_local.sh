#!/bin/bash

#Move to project directory
cd /Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT || exit


#submit to the queue
for heldout_run in {0..0}
do
  #for LR in 0.001 0.0001 0.00001
  for LR in 0.00001
  do
    for atn_heads in 4
    #for atn_heads in 3 4 5 6
    do
      # 1) $1 is the text message supplied by the user for bookkeeping
      # other arguments, in order, are
      # 2) this count in this script's loop (tells the pythons script which run to hold out as validation),
      # 3) the weight for the CLS task (between 0 and 1, "default" can be given),
      # 4) the Learning Rate ("default" can be given),
      # 5) whether to save the model at the end of training ("True" or "False" with quotes),
      # 6) pretraining task ("both", "CLS_only", "MSK_only")
      # 7) number of attention heads in the model
      # 8) number of layers in the model
      # 9) factor of forward expansion in the model
       bash timedir_launcher.sh "$1" "$heldout_run" "1" $LR "False" "CLS_only" $atn_heads "3" "4"
       sleep 2
    done
  done
done

