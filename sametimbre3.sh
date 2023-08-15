#Move to scripts directory
cd /isi/music/auditoryimagery2/seanthesis/scripts || exit

#submit to the queue
# submit finetuning jobs for models pretrained on Both
#for pti in 1 3 6 7 8 9 10 11
for pti in 8
do
  #for i in 2
  for i in {0..4}
  do
      # provide values for message, task, pretrained index, loop index, and freeze_pretrained
      sbatch sametimbre3.script "$1" "both" "$pti" "$i" "False"
      sleep 4
  done
done
