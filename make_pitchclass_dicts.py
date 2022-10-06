##### imports
import os
import sys
import audimg as A
from Constants import *
from helpers import make_pitchclass_code_dict, make_sub_info_dict, make_sub_5gram_info

###### parameters and paths
debug=1
hemisphere="left"
threshold="23"
ROI="STG"
n_runs=8 #all subjects had 8 runs in this dataset
sublist = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665", "1668", "1672", "1678", "1680"]
ROIfile_str=ROI+"_allruns"+str(hemisphere)+"_t"+str(threshold)+".p"

####### make important dictionaries
code_dict={100:"htrumpet", 200:"hclarinet", 300:"itrumpet", 400:"iclarinet", 1100:"ptrumpet"}
make_pitchclass_code_dict(targets_dir, code_dict) #targets_dir is defined in Constants.py
 #examples of above key-value pairs:
    # 152:E3 Heard Trumpet
    # 1182:A#5 Probe Trumpet

sub_info_dict={} #give the subid as dictionary-key and receive tuple containing (subidx, accession, major-key)
make_sub_info_dict(targets_dir, sub_info_dict)
if(debug):
    print(sub_info_dict)
# examples of above key-value pairs:
    # sid001401:(20,A002636,E)
    # sid001088:(27,A003000,F)

# put all the relevant info for each meaningful 5gram into a tuple
#    meaningful meaning it either contains something to decode or compare, aligned properly
#    tuples have the format (subid, accession, [i, j, k, l, m], run_n, HorI, timbre, key, pitch, octave, vividness, GoF)
#    examples of these tuples:
#    ("42", "sid001680", "A003274", [7,8,9,10,11], [11, 12, 13, 14, 15], "3", "I", "clarinet", "E", "F#", "5", 4, 4)
#    In lloyd's files, vividness and GoF is given as a float, 1.0-4.0 but is never a half value so int is fine
sub_5gram_dict = {}
make_sub_5gram_info(targets_dir, code_dict, sub_info_dict, sub_5gram_dict)



