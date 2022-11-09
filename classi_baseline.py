from Constants import *
import os
import sys

sublist = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665", "1668", "1672", "1678", "1680"]
run_str = "dt-neuro-func-task.tag-preprocessed.run-0"
for sub in sublist:
    subid="sub-sid00"+sub
    sub_path = pitchclass_preproc_path + subid+"/"+subid+"/"
    for run in range(1,9): #numbered 1 to 8
        run_dir=sub_path+run_str+str(run)+".id"
        doesexist=os.path.exists(run_dir)
        if doesexist:
            print(run_dir)
