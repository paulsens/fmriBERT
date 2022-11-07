#Move to scripts directory
cd /isi/music/auditoryimagery2/seanthesis/scripts

#submit to the queue
for i in {0..9}
do
    sbatch sametimbre1.script "$1" "$i"
    sleep 2
done
