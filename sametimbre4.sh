#Move to scripts directory
cd /isi/music/auditoryimagery2/seanthesis/scripts || exit

#submit to the queue
# submit finetuning jobs for models pretrained on binaryonly
for pti in 1 3 4 10
#for pti in 3
do
  for i in {0..4}
  #for i in 0
  do
      # provide values for message, task, pretrained index, loop index, and freeze_pretrained
      sbatch sametimbre4.script "$1" "binaryonly" "$pti" "$i" "False"
      sleep 4
  done
done
