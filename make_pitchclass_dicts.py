##### imports
import os
import sys
from Constants import *
from helpers import make_pitchclass_code_dict, make_sub_info_dict, make_sub_cycle_dict
import pickle
import random

###### parameters and paths
debug=1
hemisphere="left"
threshold="23"
ROI="STG"
n_runs=8 #all subjects had 8 runs in this dataset
sublist = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665", "1668", "1672", "1678", "1680"]
ROIfile_str=ROI+"_allruns"+str(hemisphere)+"_t"+str(threshold)+".p"

####### make important dictionaries
stim_code_dict={100:"htrumpet", 200:"hclarinet", 300:"itrumpet", 400:"iclarinet", 1100:"ptrumpet"}
make_pitchclass_code_dict(targets_dir, stim_code_dict) #targets_dir is defined in Constants.py
 #examples of above key-value pairs:
    # 152:E3 Heard Trumpet
    # 1182:A#5 Probe Trumpet
if(debug):
    print(stim_code_dict)

# dictionary keyed by subID to get subIDX, accession, and major key
sub_info_dict={} #give the subid as dictionary-key and receive tuple containing (subidx, accession, major-key)
make_sub_info_dict(targets_dir, sub_info_dict)
if(debug):
    print(sub_info_dict)
# examples of above key-value pairs:
    # sid001401:(20,A002636,E)
    # sid001088:(27,A003000,F)


# key: subid
# value: [ all tuples for this subject of this form:
# (idx, subid, accession, key, run_n, cycle_n, stimHorI, stimtimbre, stimnote, stimoctave, vividness,
# "Probe", probetimbre, probenote, probeoctave, GoF) ]
#    example of such a tuple:
#('42', 'sid001680', 'A003274', 'E', '4', '5', 'Heard', 'Clarinet', 'F#', '4', 4, 'Probe', 'Clarinet', 'B', '4', 3)
# list of cycles is in temporal order
# dictionary is passed by reference and filled in-place
sub_cycle_dict = {}
make_sub_cycle_dict(targets_dir, stim_code_dict, sub_info_dict, sub_cycle_dict)

if(debug):
    for sub in sub_info_dict.keys():
        print("For subject "+str(sub)+":")
        sub_cycles = sub_cycle_dict[sub]
        num_cycles = len(sub_cycles)
        print("Number of cycles: "+str(num_cycles))
        cycle_choice = random.randint(0, num_cycles-1) #randint is doubly inclusive for whatever reason
        print("Randomly sampled cycle-tuple: "+str(sub_cycles[cycle_choice]))

# ------ dictionaries are all created by this line

# save dicts
stim_code_file = targets_dir+"stim_code_dict.p"
sub_info_file = targets_dir+"sub_info_dict.p"
sub_cycle_file = targets_dir+"sub_cycle_dict.p"

with open(stim_code_file, "wb") as stim_code_fp:
    pickle.dump(stim_code_dict, stim_code_fp)

with open(sub_info_file, "wb") as sub_info_fp:
    pickle.dump(sub_info_dict, sub_info_fp)

with open(sub_cycle_file, "wb") as sub_cycle_fp:
    pickle.dump(sub_cycle_dict, sub_cycle_fp)

# done
print("Dictionaries written to "+str(targets_dir))





