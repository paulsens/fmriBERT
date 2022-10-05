##### imports
import os
import sys
from Constants import *
from helpers import make_pitchclass_code_dict, make_sub_info_dict, make_sub_5gram_dict
import pickle
import numpy as np
import copy

###### parameters and paths
debug=0
hemisphere="left"
threshold="23"
ROI="STG"
n_runs=8 #all subjects had 8 runs in this dataset
sublist = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665", "1668", "1672", "1678", "1680"]
ROIfile_str=ROI+"_allruns"+str(hemisphere)+"_t"+str(threshold)+".p"

####### make important dictionaries
code_dict={100:"htrumpet", 200:"hclarinet", 300:"itrumpet", 400:"iclarinet", 1100:"ptrumpet"}
make_pitchclass_code_dict(targets_dir, code_dict, debug) #targets_dir is defined in Constants.py
 #examples of above key-value pairs:
    # 152:E3 Heard Trumpet
    # 1182:A#5 Probe Trumpet

# put all the relevant info for each meaningful 5gram into a tuple
#    meaningful meaning it either contains something to decode or compare, aligned properly
#    tuples have the format (idx, subid, accession, key, cycle_n, run_n, stimHorI, stimtimbre, stimnote, stimoctave,
#                 #           probeHorI, probetimbre, probenote, probeoctave, vividness, GoF)
#    examples of these tuples:
#    ("42", "sid001680", "A003274", E, "17", "3", "Imagined", "clarinet", "E", "5",  4.0, "Probe", "clarinet", "A#", "5", 4.0)
#    In lloyd's files, vividness and GoF is given as a float, 1.0-4.0 but is never a half value so int is fine
with open(targets_dir+"sub_5gram_dict.p", "rb") as sub_fp:
    sub_5gram_dict = pickle.load(sub_fp)

with open(targets_dir+"idx_5gram_dict.p", "rb") as idx_fp:
    idx_5gram_dict = pickle.load(idx_fp)

#where is the training data?
data_path = pitchclass_preproc_path
#just put the sub-sid00xxxx in twice between these

all_samples = []
all_labels = []
voxel_dim=420
CLS = [1] + ([0] * (voxel_dim - 1))  # first dimension is reserved for cls_token flag
MSK = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
SEP = [0, 0, 1] + ([0] * (voxel_dim - 3))  # third dimension is reserved for sep_token flag

for sub in sublist:
#for sub in ["1088"]:
    subid="sub-sid00"+sub
    full_path=data_path+subid+"/"+subid+"/"+ROIfile_str
    sub_leftsamples = []
    with open(full_path, "rb") as allruns_fp:
        sub_allruns = pickle.load(allruns_fp)

    for run_n in range(0,8):
        for cycle_n in range(0, 21):
            start_TR1 = 6 + 11 * cycle_n + 233*run_n # note that if you open the text file in pycharm it starts at line 1, but .readlines() of course is a 0-indexed list, and the fMRI TR data is 0-indexed as well
            # so these numbers will match everything except opening the A00 text file in pycharm
            end_TR1 = start_TR1 + 4  # the inclusive ending TR for heard/imagined sequence
            start_TR2 = 10 + 11 * cycle_n + 233*run_n  # starting TR for probe tone sequence
            end_TR2 = start_TR2 + 4 # inclusive ending TR for probe tone sequence

            TR_list1 = list(range(start_TR1, end_TR1 + 1))
            # e.g (again these are TR numbers, line numbers in pycharm are one greater):
            # 6) 1
            # 7) 1
            # 8) 476
            # 9) 476
            # 10) 2
            TR_list2 = list(range(start_TR2, end_TR2 + 1))
            # e.g:
            # 10) 2
            # 11) 1274
            # 12) 2
            # 13) 1
            # 14) 1
            #note that TR_list 2 will go out of bounds on the very last one and should be accounted for

            this_sample = []
            this_
    # what do i want to do?
    # i want a binary classification task from the pitchclass data
    # do these two sequences have the same timbre?
    # but keep pairings within-subject
    # ok so loop through subjects,
    # the two lists are called all_samples and all_labels


