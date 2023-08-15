#Go to our virtual environment and activate it
#This environment should be python2 with mvpa and pickle
#cd /dartfs-hpc/rc/home/3/f003543/.conda/envs/mvpa3/bin
#source activate mvpa3

#Move to project directory
cd /isi/music/auditoryimagery2/seanthesis/thesis/scripts


#submit to the queue
#for i in 0
for i in {0..7}
do
   #for binweight in "0.1" "0.3" "0.5" "0.7" "0.9"
   for LR in "0.00001"
   do
      #for LR in "0.0001" "0.00005" "0.00001"
      for num_layers in 3
      do
        for atn_heads in 2
        do
          # 1) $1 is the text message supplied by the user for bookkeeping
          # other arguments, in order, are
          # 2) this count in this script's loop (tells the pythons script which run to hold out as validation),
          # 3) weight applied to CLS task when calculating loss
          # 4) the Learning Rate ("default" can be given),
          # 5) whether to save the model at the end of training ("True" or "False" with quotes),
          # 6) pretraining task ("both", "CLS_only", "MSK_only")
          # 7) number of attention heads in the model
          # 8) number of layers in the model
          # 9) factor of forward expansion in the model
          # 10) length of each sequence, either 5 or 10
          # 11) dataset, either audimg or genre
           sbatch pairedpretrain.script "$1" "$i" "1" $LR "True" "CLS_only" $atn_heads $num_layers 4 "5" "audimg"
           sleep 2
        done
    done
  done
done

