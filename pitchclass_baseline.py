
import os
import sys
import audimg as A
from Constants import *
from helpers import make_pitchclass_code_dict

hemisphere="left"
threshold="23"
n_runs=8 #all subjects had 8 runs in this dataset
sublist = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665", "1668", "1672", "1678", "1680"]
file_str="STG_allruns"+str(hemisphere)+"_t"+str(threshold)+".p"
code_dict={100:"htrumpet", 200:"hclarinet", 300:"itrumpet", 400:"iclarinet", 1100:"ptrumpet"}
make_pitchclass_code_dict(targets_dir, code_dict) #targets_dir is defined in Constants.py

for sub in sublist:
    subid="sub-sid00"+sub
    sub_path = pitchclass_preproc_path + subid+"/"+subid+"/"



