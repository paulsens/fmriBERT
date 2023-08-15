#Go to our virtual environment and activate it
#This environment should be python2 with mvpa and pickle
#cd /dartfs-hpc/rc/home/3/f003543/.conda/envs/mvpa3/bin
#source activate mvpa3

#Move to project directory
cd /isi/music/auditoryimagery2/seanthesis/scripts


#submit to the queue
sbatch pretraingpu.script "$1"


