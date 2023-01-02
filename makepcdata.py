##### imports
import os
import sys
from Constants import *
from helpers import make_pitchclass_code_dict, make_sub_info_dict, make_sub_cycle_dict, detrend_flattened, standardize_flattened
import pickle
import numpy as np
import copy
import random


#------- functions
# takes a list of indexes and partitions out a validation split
def partition_idxs(idx_list, val_size):
    train_idxs = []
    # returns a list of val_size many indexes sampled from idx_list without replacement
    val_idxs = random.sample(idx_list, val_size)
    for idx in idx_list:
        # i dont think there's a cleaner way to split up lists like this
        if idx not in val_idxs:
            train_idxs.append(idx)

    return train_idxs, val_idxs

# list of lists might be [HT_train_idxs, HC_train_idxs, IT_train_idxs, IC_train_idxs]
# we want to pair up the indices to make training samples in a specific way
# right now, each index in each list gets paired with an index from each of the four lists
# with special care to not pair an index with itself
def pair_idxs(given_lists, num_pairs=1):
    list_of_lists = []

    # if we're not including imagined data, two of the elements in given_lists will be None
    # this construction lets all the code keep working
    for given_list in given_lists:
        if(given_list) is not None:
            list_of_lists.append(given_list)

    idx_pairs = []
    for idx_list in list_of_lists: # go through each list of indexes
        for left_idx in idx_list: # each index in that list
            # store the partners selected so far
            partners_so_far = []
            # how many partners from each condition are we finding for each left_idx? default is 1
            for partner in range(0, num_pairs):
                for partner_list in list_of_lists: #pair it with an index from each list

                    if idx_list==partner_list: # if the two lists are the same, don't pair with yourself
                        right_idx=left_idx
                        while(right_idx==left_idx):
                            right_idx=random.choice(partner_list)
                            # don't pick the same one twice if we're doing more than one per condition
                            if right_idx in partners_so_far:
                                right_idx = left_idx

                    else: # otherwise, get a random selection without a care in the world
                        right_idx = random.choice(partner_list)
                        # don't pick the same one twice
                        while right_idx in partners_so_far:
                            right_idx = random.choice(partner_list)
                        # at the moment, this does allow for a right_idx to appear more than once, and thus for some to never appear on the right side, but it's such a mess to prevent that for almost no real benefit
                        # may fix that in the future

                    partners_so_far.append(right_idx)
                    pair = (left_idx, right_idx)
                    idx_pairs.append(pair)

    #now every index in every list should have been paired with an index from each of the lists
    return idx_pairs

# get a pair of indexes into a list of cycles, create the training sample and label
def get_samplelabel(pair, sub_cycles, sub_allruns, CLS, SEP, run_idx, cycle_idx, cond_idx, timbre_idx):
    left_idx = pair[0]
    right_idx = pair[1]
    left_cycle_info = sub_cycles[left_idx]
    right_cycle_info = sub_cycles[right_idx]
    subid = left_cycle_info[1]

    left_run_n = left_cycle_info[run_idx]
    left_cycle_n = left_cycle_info[cycle_idx]
    left_cond = left_cycle_info[cond_idx]
    left_timbre = left_cycle_info[timbre_idx]
    right_run_n = right_cycle_info[run_idx]
    right_cycle_n = right_cycle_info[cycle_idx]
    right_cond = right_cycle_info[cond_idx]
    right_timbre = right_cycle_info[timbre_idx]

    # dont forget that cycle_n and run_n are strings by default
    left_start_TR = 6+(11*int(left_cycle_n))+(233*int(left_run_n))
    left_end_TR = left_start_TR+5
    right_start_TR = 6+(11*int(right_cycle_n))+(233*int(right_run_n))
    right_end_TR = right_start_TR+5

    # finally create the training sample
    sample=[CLS]
    for left_i in range(left_start_TR, left_end_TR):
        sample.append(sub_allruns[left_i])

    sample.append(SEP)
    for right_i in range(right_start_TR, right_end_TR):
        sample.append(sub_allruns[right_i])


    if(left_timbre == right_timbre):
        label = True
    else:
        label = False

    # detailed labels facilitate analysis of results by keeping metadata about how we created this sample+label
    detailed_label = (label, subid, left_timbre, left_cond, left_run_n, left_cycle_n, right_timbre, right_cond, right_run_n, right_cycle_n)

    return sample, detailed_label









###### parameters and paths
debug=1
hemisphere="left"
threshold="23"
ROI="STG"
n_runs=8 #all subjects had 8 runs in this dataset
sublist = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665", "1668", "1672", "1678", "1680"]
ROIfile_str=ROI+"_allruns"+str(hemisphere)+"_t"+str(threshold)+".p"

####### load important dictionaries, made by make_pitchclass_dicts.py
# targets_dir is imported from Constants.py and depends on "env" variable (also defined in Constants.py)

# first is the code_dict
# examples of key-value pairs (note both keys and values are strings):
# "152":"E3 Heard Trumpet"
# "1182":"A#5 Probe Trumpet"
with open(targets_dir+"stim_code_dict.p", "rb") as stim_code_dict_fp:
    stim_code_dict = pickle.load(stim_code_dict_fp)


# second is the sub_info_dict
# key: subid
# value: (sub-idx, accession, majorkey)
# examples of key-value pairs (note all elements of the tuples are strings):
    # "sid001401":("20","A002636","E")
    # "sid001088":("27","A003000","F")
with open(targets_dir+"sub_info_dict.p", "rb") as sub_info_fp:
    sub_info_dict = pickle.load(sub_info_fp)

# third is the sub_cycle_dict
# key: subid
# value: [ all tuples for this subject of this form:
# (idx, subid, accession, key, run_n, cycle_n, stimHorI, stimtimbre, stimnote, stimoctave, vividness,
# "Probe", probetimbre, probenote, probeoctave, GoF) ]
#    example of such a tuple:
# ('42', 'sid001680', 'A003274', 'E', '4', '5', 'Heard', 'Clarinet', 'F#', '4', 4, 'Probe', 'Clarinet', 'B', '4', 3)
# list of cycles is in temporal order
with open(targets_dir+"sub_cycle_dict.p", "rb") as sub_cycle_fp:
    sub_cycle_dict = pickle.load(sub_cycle_fp)

# dictionary for getting the index of whatever you want in the above tuples
info_idx = {
    "sub_idx":0,
    "sub_id":1,
    "accession":2,
    "majorkey":3,
    "run_n":4,
    "cycle_n":5,
    "stim_cond":6,
    "stim_timbre":7,
    "stim_note":8,
    "stim_octave":9,
    "vividness":10,
    "probe":11,
    "probe_timbre":12,
    "probe_note":13,
    "probe_octave":14,
    "GoF":15
}

#------ dictionaries are all loaded at this point


#where is the training data?
data_path = pitchclass_preproc_path
save_path = "/Volumes/External/pitchclass/finetuning/sametimbre/datasets/8/"
#just put the sub-sid00xxxx in twice between these

all_X = []
all_y = []
all_val_X = []
all_val_y = []

voxel_dim=420
CLS = [1] + ([0] * (voxel_dim - 1))  # first dimension is reserved for cls_token flag
MSK = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
SEP = [0, 0, 1] + ([0] * (voxel_dim - 3))  # third dimension is reserved for sep_token flag

# ------- at the moment we're only creating within-subject pairs, but the above global lists will hold samples from all subjects
if __name__ == "__main__":
    NUM_TOKENS=2 # number of dimensions reserved by tokens, e.g CLS/MSK
    MAKE_PAIRS = False # are we pairing two sequences or is each input a single sequence
    do_subjects = True
    do_all = True
    include_imagined = True
    # do we want to partition out some percentage of the samples or do we want to hold out runs?
    holdout_runs = False
    # do_subjects will create and pickle each subject separately
    # do_all assumes those subject pickled files already exist, so you can run both at once or do_subjects first then do_all
    if do_subjects:
        #for sub in ["1088"]:
        for sub in sublist:
            truecount=0
            falsecount=0
            valtruecount=0
            valfalsecount=0
            sub_X = []
            sub_y = []
            sub_val_X = []
            sub_val_y = []

            subid="sub-sid00"+sub
            shortsubid="sid00"+sub
            full_path=data_path+subid+"/"+subid+"/"+ROIfile_str
            with open(full_path, "rb") as allruns_fp:
                sub_allruns = pickle.load(allruns_fp)
            xlen = len(sub_allruns)
            ylen = len(sub_allruns[0])
            # print("some samples before detrending: ")
            # print(sub_allruns[100])
            # print(sub_allruns[1000])
            sub_allruns = detrend_flattened(sub_allruns, detrend="linear", num_tokens=NUM_TOKENS)
            # print("some samples after detrending: ")
            # print(sub_allruns[100])
            # print(sub_allruns[1000])
            xlen = len(sub_allruns)
            ylen = len(sub_allruns[0])
            #print("before standardize, shape is " + str(xlen) + "," + str(ylen))

            sub_allruns = standardize_flattened(sub_allruns, num_tokens=NUM_TOKENS)
            xlen = len(sub_allruns)
            ylen = len(sub_allruns[0])
            #print("after standardize, shape is " + str(xlen) + "," + str(ylen))
            #print("some examples are: \n")
            #print(str(sub_allruns[100]))
            #print(str(sub_allruns[1000]))
            # each element of sub_cycle has all relevant information for a cycle for this subject
            sub_cycles=sub_cycle_dict[shortsubid]

            # there may be some way to abstract this all out but for now let's just hardcode the training data
            sample = []
            label = []
            # what info are we looking for?
            cond_idx = info_idx["stim_cond"] #the stimulus condition, Heard or Imagined
            timbre_idx = info_idx["stim_timbre"] # either Trumpet or CLarinet
            cycle_idx = info_idx["cycle_n"]
            run_idx = info_idx["run_n"]
            HT_idxs = []
            HC_idxs = []
            IT_idxs = []
            IC_idxs = []
            HT_train_idxs = []
            HC_train_idxs = []
            HT_val_idxs = []
            HC_val_idxs = []
            IT_train_idxs = []
            IC_train_idxs = []
            IT_val_idxs = []
            IC_val_idxs = []
            # we use the above indices to check/retrieve information we want from the long tuples in sub_cycles

            # holdout_runs means hold out entire runs, this path instead holds out some percentage of cycles
            if not holdout_runs:
                for idx in range(0, len(sub_cycles)):
                    #print("len of sub_cycles is "+str(len(sub_cycles)))
                    cycle_info = sub_cycles[idx]
                    cycle_n = cycle_info[cycle_idx]
                    run_n = cycle_info[run_idx]
                    cond = cycle_info[cond_idx]
                    timbre = cycle_info[timbre_idx]

                    if(cond=="Heard"):
                        if(timbre=="Clarinet"):
                            HC_idxs.append(idx)
                        elif(timbre=="Trumpet"):
                            HT_idxs.append(idx)

                    elif (cond=="Imagined") and include_imagined:
                        if(timbre=="Clarinet"):
                            IC_idxs.append(idx)
                        elif(timbre=="Trumpet"):
                            IT_idxs.append(idx)

                # so now the four lists have all their corresponding indexes into sub_cycles
                # split them up into training and validation
                HC_train_idxs, HC_val_idxs = partition_idxs(HC_idxs, 6)
                HT_train_idxs, HT_val_idxs = partition_idxs(HT_idxs, 6)
                if include_imagined:
                    IC_train_idxs, IC_val_idxs = partition_idxs(IC_idxs, 6)
                    IT_train_idxs, IT_val_idxs = partition_idxs(IT_idxs, 6)
                else:
                    IC_train_idxs, IC_val_idxs = None, None
                    IT_train_idxs, IT_val_idxs = None, None
                # print("HC_train_idxs are "+str(HC_train_idxs))
                # print("HT_train_idxs are "+str(HT_train_idxs))
                # print("IC_train_idxs are "+str(IC_train_idxs))
                # print("IT_train_idxs are "+str(IT_train_idxs))

            # if we are holding out entire runs
            else:
                for idx in range(0, len(sub_cycles)):
                    #print("len of sub_cycles is "+str(len(sub_cycles)))
                    cycle_info = sub_cycles[idx]
                    cycle_n = cycle_info[cycle_idx]
                    run_n = cycle_info[run_idx]
                    cond = cycle_info[cond_idx]
                    timbre = cycle_info[timbre_idx]

                    if(cond=="Heard"):
                        if(timbre=="Clarinet"):
                            # runs start at 0, 4-7 are the second half
                            if(int(run_n)<4):
                                HC_train_idxs.append(idx)
                            else:
                                HC_val_idxs.append(idx)
                        elif(timbre=="Trumpet"):
                            if(int(run_n)<4):
                                HT_train_idxs.append(idx)
                            else:
                                HT_val_idxs.append(idx)

                    elif (cond=="Imagined") and include_imagined:
                        if(timbre=="Clarinet"):
                            if(int(run_n)<4):
                                IC_train_idxs.append(idx)
                            else:
                                IC_val_idxs.append(idx)
                        elif(timbre=="Trumpet"):
                            if(int(run_n)<4):
                                IT_train_idxs.append(idx)
                            else:
                                IT_val_idxs.append(idx)
                if not include_imagined:
                    IC_train_idxs, IC_val_idxs = None, None
                    IT_train_idxs, IT_val_idxs = None, None
                #print("for subject "+str(sub)+", lengths are "+str(len(HC_train_idxs))+", "+str(len(HT_train_idxs))+", "+str(len(IC_train_idxs))+", "+str(len(IT_train_idxs))+", "+str(len(HC_val_idxs))+", "+str(len(HT_val_idxs))+", "+str(len(IC_val_idxs))+", "+str(len(IT_val_idxs))+", ")



            # print("trainidx_pairs is "+str(trainidx_pairs))
            # print("validx_pairs is "+str(validx_pairs))
            # break
            # finally create training data for this subject
            if MAKE_PAIRS:
                trainidx_pairs = pair_idxs([HC_train_idxs, HT_train_idxs, IC_train_idxs, IT_train_idxs], num_pairs=4)
                validx_pairs = pair_idxs([HC_val_idxs, HT_val_idxs, IC_val_idxs, IT_val_idxs], num_pairs=1)
                for pair in trainidx_pairs:
                    # pair is two indexes into sub_cycles
                    # function returns a sample like [CLS token token token token SEP token ... ]
                    # and a label either True or False, for right now
                    sample, label = get_samplelabel(pair, sub_cycles, sub_allruns, CLS, SEP, run_idx, cycle_idx, cond_idx, timbre_idx)
                    if(label[0]==True):
                        truecount+=1
                    else:
                        falsecount+=1
                    sub_X.append(sample)
                    #print("appended "+str(sample)+" to sub_X")
                    sub_y.append(label)
                    #print("appended "+str(label)+" to sub_y")


                for valpair in validx_pairs:
                    valsample, vallabel = get_samplelabel(valpair, sub_cycles, sub_allruns, CLS, SEP, run_idx, cycle_idx, cond_idx, timbre_idx)
                    if(vallabel[0]==True):
                        valtruecount+=1
                    else:
                        valfalsecount+=1
                    sub_val_X.append(valsample)
                    sub_val_y.append(vallabel)
                print("True and false count for subject "+str(shortsubid)+" were "+str(truecount)+", "+str(falsecount))
                print("Validation split True and false count for subject " + str(shortsubid) + " were " + str(valtruecount) + ", " + str(valfalsecount))

            # save the training data for just this subject in case we want it later
            sub_save_path = save_path+shortsubid+"_"
            with open(sub_save_path+"X.p", "wb") as temp_fp:
                pickle.dump(sub_X, temp_fp)
            with open(sub_save_path+"y.p", "wb") as temp_fp:
                pickle.dump(sub_y, temp_fp)
            with open(sub_save_path+"val_X.p", "wb") as temp_fp:
                pickle.dump(sub_val_X, temp_fp)
            with open(sub_save_path+"val_y.p", "wb") as temp_fp:
                pickle.dump(sub_val_y, temp_fp)

            print("Saved training data for subject "+shortsubid)

    if do_all:
        all_X = []
        all_y = []
        all_val_X = []
        all_val_y = []
        # load each subjects's contributions
        for sub in sublist:
            shortsubid = "sid00"+sub

            sub_save_path = save_path+shortsubid+"_"
            with open(sub_save_path+"X.p", "rb") as temp_fp:
                sub_X = pickle.load(temp_fp)
            with open(sub_save_path+"y.p", "rb") as temp_fp:
                sub_y = pickle.load(temp_fp)
            with open(sub_save_path+"val_X.p", "rb") as temp_fp:
                sub_val_X = pickle.load(temp_fp)
            with open(sub_save_path+"val_y.p", "rb") as temp_fp:
                sub_val_y = pickle.load(temp_fp)

            # append this subject's training data to the global training data
            all_X.extend(sub_X)
            all_y.extend(sub_y)
            all_val_X.extend(sub_val_X)
            all_val_y.extend(sub_val_y)
            print("extended "+str(shortsubid))

        print("all_X has length"+str(len(all_X)))
        print("all_y has length"+str(len(all_y)))
        print("all_val_X has length"+str(len(all_val_X)))
        print("all_val_y has length"+str(len(all_val_y)))

        all_save_path = save_path + "all_"
        with open(all_save_path + "X.p", "wb") as temp_fp:
            pickle.dump(all_X, temp_fp)
        with open(all_save_path + "y.p", "wb") as temp_fp:
            pickle.dump(all_y, temp_fp)
        with open(all_save_path + "val_X.p", "wb") as temp_fp:
            pickle.dump(all_val_X, temp_fp)
        with open(all_save_path + "val_y.p", "wb") as temp_fp:
            pickle.dump(all_val_y, temp_fp)
        print("Saved concatenated training data for all subjects")





