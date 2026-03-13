#Go to our virtual environment and activate it
#This environment should be python2 with mvpa and pickle
#cd /dartfs-hpc/rc/home/3/f003543/.conda/envs/mvpa3/bin
#source activate mvpa3

#Move to project directory
cd /isi/music/auditoryimagery2/seanthesis/scripts


#submit to the queue
#for i in {0..11}
for i in 0
do
   #for binweight in "0.1" "0.3" "0.5" "0.7" "0.9"
   for binweight in "0.1"
   do
      #for lr in "0.0001" "0.00005" "0.00001"
      for lr in "0.0001"
      do
         sbatch pretrain1.script "$1" "$i" "$binweight" "$lr"
         sleep 4
      done
   done
done

