#Go to our virtual environment and activate it
#This environment should be python2 with mvpa and pickle
#cd /dartfs-hpc/rc/home/3/f003543/.conda/envs/mvpa3/bin
#source activate mvpa3

#Move to project directory
cd /isi/music/auditoryimagery2/seanthesis/thesis/scripts


#submit to the queue
for i in {0..7}
#for i in 10 11
do
    #for LR in "0.0001" "0.00005" "0.00001"
    for LR in "0.00001"
    do
      for atn_heads in 2
      do
        # 1) $1 is the text message supplied by the user for bookkeeping
        # other arguments, in order, are
        # 2) the run that was held out during pretraining, directs model to finetuning dataset and saved model,
        # 3) whether to freeze the weights of the pretrained model, ("True" or "False")
        # 4) the Learning Rate ("default" can be given),
        # 5) whether to save the model at the end of training ("True" or "False" with quotes),
        # 6) pretraining task ("both", "CLS_only", "MSK_only", "fresh", "leftHeldoutSubs/Fresh", "rightHeldoutSubs/Fresh")
        # 7) number of attention heads in the model
        # 8) number of layers in the model
        # 9) factor of forward expansion in the model
        # 10) length of each sequence, either 5 or 10
        # 11) finetune task, either sametimbre or samegenre
        # 12) iteration in the outer loop, not necessarily the same as the heldout run index, especially if we're crossing datasets or doing hpsearch
        # 13) Saved model index
         sbatch pairedfinetune.script "$1" "$i" "False" $LR "False" "leftHeldoutSubs" $atn_heads "3" "4" "5" "sametimbre" $i $i
         sleep 2
      done
  done
done

