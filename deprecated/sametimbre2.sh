#!/bin/bash
#Move to scripts directory
cd /isi/music/auditoryimagery2/seanthesis/scripts

#submit to the queue
export pti=5
for i in {0..9}
do
    sbatch sametimbre2.script "$1" "both" "$pti" "$i" "False"
    sleep 4
done
