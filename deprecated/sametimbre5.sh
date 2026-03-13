#Move to scripts directory
cd /isi/music/auditoryimagery2/seanthesis/scripts || exit

#submit to the queue
# submit finetuning jobs for models pretrained on multionly
for pti in 1 3 5 9
do
  for i in {0..4}
  do
      # provide values for message, task, pretrained index, loop index, and freeze_pretrained
      sbatch sametimbre5.script "$1" "multionly" "$pti" "$i" "False"
      sleep 4
  done
done
