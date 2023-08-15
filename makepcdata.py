##### imports
import os
import sys
from Constants import *
from helpers import make_pitchclass_code_dict, make_sub_info_dict, make_sub_cycle_dict, detrend_flattened, standardize_flattened
import pickle
import numpy as np
import copy
import random
from datetime import date


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
#repetitions is either 1, 2 or 4. For 1, each leftsample will be paired with either a positive or negative parter of the same condition. 2 means each leftsample will be paired with positive and negative partner with same condition. 4 means each leftsample will be paired with a positive and negative sample of both conditions. It seems that repetition of 4 can lead to overfitting.

def pair_idxs(given_lists, num_pairs=1, repetitions=2):
    list_of_lists = []
    # lists is order HC HT IC IT
    HC_idx = 0
    HC_list = given_lists[HC_idx]
    HT_idx = 1
    HT_list = given_lists[HT_idx]
    IC_idx = 2
    IC_list = given_lists[IC_idx]
    IT_idx = 3
    IT_list = given_lists[IT_idx]

    # if we're not including imagined data, two of the elements in given_lists will be None
    # this construction lets all the code keep working
    for given_list in given_lists:
        if(given_list) is not None:
            list_of_lists.append(given_list)

    idx_pairs = []

    if repetitions == 4:
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

    elif repetitions==2:
        heard_lists = [HC_list, HT_list]
        for idx_list in heard_lists:  # go through each list of indexes
            for left_idx in idx_list:  # each index in that list
                # store the partners selected so far
                partners_so_far = []
                # how many partners from each condition are we finding for each left_idx? default is 1
                for partner in range(0, num_pairs):
                    for partner_list in heard_lists:  # pair it with an index from each list

                        if idx_list == partner_list:  # if the two lists are the same, don't pair with yourself
                            right_idx = left_idx
                            while (right_idx == left_idx):
                                right_idx = random.choice(partner_list)
                                # don't pick the same one twice if we're doing more than one per condition
                                if right_idx in partners_so_far:
                                    right_idx = left_idx

                        else:  # otherwise, get a random selection without a care in the world
                            right_idx = random.choice(partner_list)
                            # don't pick the same one twice
                            while right_idx in partners_so_far:
                                right_idx = random.choice(partner_list)
                            # at the moment, this does allow for a right_idx to appear more than once, and thus for some to never appear on the right side, but it's such a mess to prevent that for almost no real benefit
                            # may fix that in the future

                        partners_so_far.append(right_idx)
                        pair = (left_idx, right_idx)
                        idx_pairs.append(pair)

        imagined_lists = [IC_list, IT_list]
        for idx_list in imagined_lists:  # go through each list of indexes
            for left_idx in idx_list:  # each index in that list
                # store the partners selected so far
                partners_so_far = []
                # how many partners from each condition are we finding for each left_idx? default is 1
                for partner in range(0, num_pairs):
                    for partner_list in imagined_lists:  # pair it with an index from each list

                        if idx_list == partner_list:  # if the two lists are the same, don't pair with yourself
                            right_idx = left_idx
                            while (right_idx == left_idx):
                                right_idx = random.choice(partner_list)
                                # don't pick the same one twice if we're doing more than one per condition
                                if right_idx in partners_so_far:
                                    right_idx = left_idx

                        else:  # otherwise, get a random selection without a care in the world
                            right_idx = random.choice(partner_list)
                            # don't pick the same one twice
                            while right_idx in partners_so_far:
                                right_idx = random.choice(partner_list)
                            # at the moment, this does allow for a right_idx to appear more than once, and thus for some to never appear on the right side, but it's such a mess to prevent that for almost no real benefit
                            # may fix that in the future

                        partners_so_far.append(right_idx)
                        pair = (left_idx, right_idx)
                        idx_pairs.append(pair)

    elif repetitions==1:
        heard_lists = [HC_list, HT_list]
        for idx_list in heard_lists:  # go through each list of indexes
            for left_idx in idx_list:  # each index in that list
                # store the partners selected so far
                partners_so_far = []

                # how many partners from each condition are we finding for each left_idx? default is 1
                for partner in range(0, num_pairs):
                    posneg_choice = random.choice([0, 1])
                    partner_list = heard_lists[posneg_choice]

                    if idx_list == partner_list:  # if the two lists are the same, don't pair with yourself
                        right_idx = left_idx
                        while (right_idx == left_idx):
                            right_idx = random.choice(partner_list)
                            # don't pick the same one twice if we're doing more than one per condition
                            if right_idx in partners_so_far:
                                right_idx = left_idx

                    else:  # otherwise, get a random selection without a care in the world
                        right_idx = random.choice(partner_list)
                        # don't pick the same one twice
                        while right_idx in partners_so_far:
                            right_idx = random.choice(partner_list)
                        # at the moment, this does allow for a right_idx to appear more than once, and thus for some to never appear on the right side, but it's such a mess to prevent that for almost no real benefit
                        # may fix that in the future

                    partners_so_far.append(right_idx)
                    pair = (left_idx, right_idx)
                    idx_pairs.append(pair)

        imagined_lists = [IC_list, IT_list]
        for idx_list in imagined_lists:  # go through each list of indexes
            for left_idx in idx_list:  # each index in that list
                # store the partners selected so far
                partners_so_far = []
                # how many partners from each condition are we finding for each left_idx? default is 1
                for partner in range(0, num_pairs):
                    posneg_choice = random.choice([0,1])
                    partner_list = imagined_lists[posneg_choice]

                    if idx_list == partner_list:  # if the two lists are the same, don't pair with yourself
                        right_idx = left_idx
                        while (right_idx == left_idx):
                            right_idx = random.choice(partner_list)
                            # don't pick the same one twice if we're doing more than one per condition
                            if right_idx in partners_so_far:
                                right_idx = left_idx

                    else:  # otherwise, get a random selection without a care in the world
                        right_idx = random.choice(partner_list)
                        # don't pick the same one twice
                        while right_idx in partners_so_far:
                            right_idx = random.choice(partner_list)
                        # at the moment, this does allow for a right_idx to appear more than once, and thus for some to never appear on the right side, but it's such a mess to prevent that for almost no real benefit
                        # may fix that in the future

                    partners_so_far.append(right_idx)
                    pair = (left_idx, right_idx)
                    idx_pairs.append(pair)

    else:
        print("illegal value for repetitions, must 1 2 or 4, quitting...")
        quit(0)
    #now every index in every list should have been paired with an index from each of the lists

    return idx_pairs

# get a pair of indexes into a list of cycles, create the training sample and label
def get_samplelabel(pair, sub_cycles, sub_allruns, CLS, SEP, run_idx, cycle_idx, cond_idx, timbre_idx, seq_len):
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

    cyclesperrun = 21
    TRsperrun = 233
    # TRs are zero indexed, first stimulus is on TR 8 (line 9 in the text file)
    # HRF peaks after 6.5 seconds or 3 TRs, so it probably makes the most sense to do STIM STIM REST PROBE REST since the probe is always the same timbre
    startTR = 8
    TRs_between_cycles = 11

    # dont forget that cycle_n and run_n are strings by default
    left_start_TR = startTR+(TRs_between_cycles*int(left_cycle_n))+(TRsperrun*int(left_run_n))
    left_end_TR = left_start_TR+seq_len
    right_start_TR = startTR+(TRs_between_cycles*int(right_cycle_n))+(TRsperrun*int(right_run_n))
    right_end_TR = right_start_TR+seq_len

    # finally create the training sample
    sample=[copy.deepcopy(CLS)]
    #print("after appending CLS, sample looks like "+str(sample))
    for left_i in range(left_start_TR, left_end_TR):
        sample.append(sub_allruns[left_i])
    #print("after appending leftsample, sample looks like "+str(sample))

    sample.append(copy.deepcopy(SEP))
    for right_i in range(right_start_TR, right_end_TR):
        sample.append(sub_allruns[right_i])
    #print("after appending rightsample, second half of sample looks like "+str(sample[6:]))


    if(left_timbre == right_timbre):
        label = True
    else:
        label = False

    # detailed labels facilitate analysis of results by keeping metadata about how we created this sample+label
    detailed_label = (label, subid, left_timbre, left_cond, left_run_n, left_cycle_n, right_timbre, right_cond, right_run_n, right_cycle_n)

    return sample, detailed_label

# get a pair of indexes into a list of cycles, create the training sample and label
# different from get_samplelabel, doesn't do pairs of subsequences, but rather just a single subsequence for timbre decoding
def get_samplelabel_single(this_idx, sub_cycles, sub_allruns, CLS, run_idx, cycle_idx, cond_idx, timbre_idx, majorkey_idx, stimnote_idx, stimoctave_idx, vividness_idx, GoF_idx, sample_length=None, do_timedir=False):

    global forward_count, reverse_count

    cycle_info = sub_cycles[this_idx]
    subid = cycle_info[1]

    run_n = cycle_info[run_idx]
    cycle_n = cycle_info[cycle_idx]
    cond = cycle_info[cond_idx]
    timbre = cycle_info[timbre_idx]
    majorkey = cycle_info[majorkey_idx]
    stimnote = cycle_info[stimnote_idx]
    stimoctave = cycle_info[stimoctave_idx]
    vividness = cycle_info[vividness_idx]
    GoF = cycle_info[GoF_idx]





    # dont forget that cycle_n and run_n are strings by default
    if sample_length == 5:
        start_TR = 6+(11*int(cycle_n))+(233*int(run_n))
        end_TR = start_TR+5
    elif sample_length == 10:
        start_TR = 5+(11*int(cycle_n))+(233*int(run_n))
        end_TR = start_TR+10
    else:
        print("Illegal sequence length, must be 5 or 10, quitting...")
        quit(0)

    # timedir stuff
    choice = 0 # by default we're not reversing stuff
    incr = 1 # for loop increments from left to right if it's not reversed
    # are we reversing stuff?
    if do_timedir:
        choice = random.choice([0,1])
        # 1 means we are reversing
        if choice == 1:
            temp = end_TR-1
            end_TR = start_TR-1 # e.g 0 to 10 becomes 9 to -1, because the right bound is excluded from range()
            start_TR = temp
            incr = -1
            reverse_count+=1

        else:
            forward_count+=1
        # if choice==0, everything is already fine

    # finally create the training sample
    sample=[CLS]
    for i in range(start_TR, end_TR, incr):
        if i>=len(sub_allruns):
            sample.append(sub_allruns[len(sub_allruns)-1]) # we don't have enough rest state to pad the final sample from each run. One option is to drop this entirely, but that's 8*17=136 samples in the trash, which is a lot. This setup just repeats the last TR of each run as padding, either at the beginning (if reversed) or at the end of the sample. Something to think about improving.
        else:
            sample.append(sub_allruns[i])

    # detailed labels facilitate analysis of results by keeping metadata about how we created this sample+label
    detailed_label = (timbre, cond, subid, run_n, cycle_n, majorkey, stimnote, stimoctave, vividness, GoF, choice)
    #print("detailed label is "+str(detailed_label))


    return sample, detailed_label







###### parameters and paths
debug=1
NUM_TOKENS = 2  # number of dimensions reserved by tokens, e.g CLS/MSK
hemisphere="right"
threshold="23"
ROI="STG"
n_runs=8 #all subjects had 8 runs in this dataset
sublist = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665", "1668", "1672", "1678", "1680"]
if NUM_TOKENS == 2:
    ROIfile_str=ROI+"_allruns"+str(hemisphere)+"_t"+str(threshold)+"_"+str(NUM_TOKENS)+"tokens.p"
else:
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


#where is the original data loaded from?
data_path = pitchclass_preproc_path
# save training_samples and training_labels
time = date.today()
seq_len = 5
forward_count=0
reverse_count=0
#save_path = "/Volumes/External/timedir/training_data/"+str(time)+"-"+str(seq_len)+"TR_audimg_HOruns_hasimagined/"
save_path = "/Volumes/External/pitchclass/finetuning/sametimbre/datasets/"+str(time)+"-"+str(seq_len)+"TR_audimg_HOruns_hasimagined_1-1-pairs_repetition1_rightSTG_runsheldout/"

if not os.path.exists(save_path):
    os.mkdir(save_path)

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
    MAKE_PAIRS = True # are we pairing two sequences or is each input a single sequence
    do_subjects = True
    do_all = True
    include_imagined = True
    check_vividness = False
    do_timedir = False
    check_GoF = False
    cross_decode = None # HI, IH, or None (first letter is training, second is validation)
    # do we want to partition out some percentage of the samples or do we want to hold out runs?
    holdout_runs = True
    holdout_subjects = None # either none or num:amount. num is how many subjects each fold should hold out, amount is how many of their possible validation runs should be held out, either half or all. For example, if we're cross decoding HI, only imagined runs may be held out for validation, of which there are 4, so either half or all of those might be held out. If we're, say, decoding timbre on both heard and imagined, there are 8 possible runs for each subject to hold out, so we might hold out 8 or 4. This abstraction of "half or all" allows us to assign the other parameters of the dataset without worrying about the exact explicit number of heldout runs.
    # do_subjects will create and pickle each subject separately
    # do_all assumes those subject pickled files already exist, so you can run both at once or do_subjects first then do_all
    if do_subjects:
        #for sub in ["1088"]:
        for sub in sublist:
            clarinet_count=0
            trumpet_count=0
            heard_count=0
            imagined_count=0
            hc_count=0
            ht_count=0
            ic_count=0
            it_count=0

            val_clarinet_count=0
            val_trumpet_count=0
            val_heard_count=0
            val_imagined_count=0
            val_hc_count=0
            val_ht_count=0
            val_ic_count=0
            val_it_count=0

            # these are only used when MAKE_PAIRS
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
            # print("some samples after standardizing: ")
            # print(sub_allruns[100])
            # print(sub_allruns[1000])

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
            majorkey_idx = info_idx["majorkey"]
            stimnote_idx = info_idx["stim_note"]
            stimoctave_idx = info_idx["stim_octave"]
            vividness_idx = info_idx["vividness"]
            GoF_idx = info_idx["GoF"]

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

            # if we're doing cross decoding, send either all heard samples to train lists and imagined to val lists or vice versa
            if cross_decode is not None:
                # if we train on Heard test on Imagined
                if cross_decode == "HI":
                    train_cond = "Heard"
                    val_cond = "Imagined"

                    train_C_idxs = HC_train_idxs
                    train_T_idxs = HT_train_idxs

                    val_C_idxs = IC_val_idxs
                    val_T_idxs = IT_val_idxs

                # train on Imagined test on Heard
                elif cross_decode == "IH":
                    train_cond = "Imagined"
                    val_cond = "Heard"

                    train_C_idxs = IC_train_idxs
                    train_T_idxs = IT_train_idxs

                    val_C_idxs = HC_val_idxs
                    val_T_idxs = HT_val_idxs

                for idx in range(0, len(sub_cycles)):
                    #print("len of sub_cycles is "+str(len(sub_cycles)))
                    # get information pertaining to this cycle
                    cycle_info = sub_cycles[idx]
                    cycle_n = cycle_info[cycle_idx]
                    run_n = cycle_info[run_idx]
                    cond = cycle_info[cond_idx]
                    timbre = cycle_info[timbre_idx]
                    majorkey = cycle_info[majorkey_idx]
                    stimnote = cycle_info[stimnote_idx]
                    stimoctave = cycle_info[stimoctave_idx]
                    vividness = cycle_info[vividness_idx]
                    # if we care about vividness, skip any cycle where no response was given for vividness
                    if check_vividness and vividness==0:
                        continue
                    GoF = cycle_info[GoF_idx]
                    # if we care about GoF, skip any cycle where no response was given for GoF
                    if check_GoF and GoF==0:
                        continue

                    if cond == train_cond:
                        if timbre=="Clarinet":
                            train_C_idxs.append(idx)
                        elif timbre=="Trumpet":
                            train_T_idxs.append(idx)
                    elif cond == val_cond:
                        if timbre=="Clarinet":
                            val_C_idxs.append(idx)
                        elif timbre=="Trumpet":
                            val_T_idxs.append(idx)

                print("With cross decoding set to "+str(cross_decode)+", HC_train has "+str(len(HC_train_idxs))+
                      ", HT_train has "+str(len(HT_train_idxs))+", IC_train has "+str(len(IC_train_idxs))+", IT_train has "+str(len(IT_train_idxs))+", HC_val has "+str(len(HC_val_idxs))+", HT_val has "+str(len(HT_val_idxs))+", IC_val has "+str(len(IC_val_idxs))+", and IT_val has "+str(len(IT_val_idxs)))


            # holdout_runs means hold out entire runs, this path instead holds out some percentage of cycles
            elif not holdout_runs:
                for idx in range(0, len(sub_cycles)):
                    #print("len of sub_cycles is "+str(len(sub_cycles)))
                    cycle_info = sub_cycles[idx]
                    cycle_n = cycle_info[cycle_idx]
                    run_n = cycle_info[run_idx]
                    cond = cycle_info[cond_idx]
                    timbre = cycle_info[timbre_idx]
                    majorkey = cycle_info[majorkey_idx]
                    stimnote = cycle_info[stimnote_idx]
                    stimoctave = cycle_info[stimoctave_idx]
                    vividness = cycle_info[vividness_idx]
                    if check_vividness and vividness == 0:
                        continue
                    GoF = cycle_info[GoF_idx]
                    # if we care about GoF, skip any cycle where no response was given for GoF
                    if check_GoF and GoF == 0:
                        continue

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

                # so now the four lists have all their corresponding indexes wrt sub_cycles
                # split them up into training and validation
                HC_train_idxs, HC_val_idxs = partition_idxs(HC_idxs, 6)
                HT_train_idxs, HT_val_idxs = partition_idxs(HT_idxs, 6)
                if include_imagined:
                    partition_size = 6
                    if len(IC_idxs) < partition_size:
                        partition_size = 1
                    IC_train_idxs, IC_val_idxs = partition_idxs(IC_idxs, partition_size)
                    partition_size = 6
                    if len(IT_idxs) < partition_size:
                        partition_size = 1
                    IT_train_idxs, IT_val_idxs = partition_idxs(IT_idxs, partition_size)
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
                    majorkey = cycle_info[majorkey_idx]
                    stimnote = cycle_info[stimnote_idx]
                    stimoctave = cycle_info[stimoctave_idx]
                    vividness = cycle_info[vividness_idx]
                    if check_vividness and vividness == 0:
                        continue
                    GoF = cycle_info[GoF_idx]
                    # if we care about GoF, skip any cycle where no response was given for GoF
                    if check_GoF and GoF == 0:
                        continue

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
                trainidx_pairs = pair_idxs([HC_train_idxs, HT_train_idxs, IC_train_idxs, IT_train_idxs], num_pairs=1, repetitions=1)
                validx_pairs = pair_idxs([HC_val_idxs, HT_val_idxs, IC_val_idxs, IT_val_idxs], num_pairs=1, repetitions=1)
                for pair in trainidx_pairs:
                    # pair is two indexes into sub_cycles
                    # function returns a sample like [CLS token token token token SEP token ... ]
                    # and a label either True or False, for right now
                    sample, label = get_samplelabel(pair, sub_cycles, sub_allruns, CLS, SEP, run_idx, cycle_idx, cond_idx, timbre_idx, seq_len)
                    if(label[0]==True):
                        truecount+=1
                    else:
                        falsecount+=1
                    sub_X.append(sample)
                    #print("appended "+str(sample)+" to sub_X")
                    sub_y.append(label)
                    #print("appended "+str(label)+" to sub_y")


                for valpair in validx_pairs:
                    valsample, vallabel = get_samplelabel(valpair, sub_cycles, sub_allruns, CLS, SEP, run_idx, cycle_idx, cond_idx, timbre_idx, seq_len)
                    if(vallabel[0]==True):
                        valtruecount+=1
                    else:
                        valfalsecount+=1
                    sub_val_X.append(valsample)
                    sub_val_y.append(vallabel)
                print("True and false count for subject "+str(shortsubid)+" were "+str(truecount)+", "+str(falsecount))
                print("Validation split True and false count for subject " + str(shortsubid) + " were " + str(valtruecount) + ", " + str(valfalsecount))

            # if not MAKE_PAIRS

            else:
                # give it the old ocular patdown
                print([HC_train_idxs, HT_train_idxs, IC_train_idxs, IT_train_idxs])
                print([HC_val_idxs, HT_val_idxs, IC_val_idxs, IT_val_idxs])

                # now fill sub_X sub_y sub_val_X and sub_val_y

                # create Heard Clarinet training data
                # detailed label is (timbre, cond, subid, run_n, cycle_n, majorkey, stimnote, stimoctave, vividness, GoF)
                # e.g ('Clarinet', 'Heard', 'sid001088', '0', '0', 'F', 'D', '5', 0, 3)

                for HC_train_idx in HC_train_idxs:
                    sample, label = get_samplelabel_single(HC_train_idx, sub_cycles, sub_allruns, CLS, run_idx, cycle_idx, cond_idx, timbre_idx, majorkey_idx, stimnote_idx, stimoctave_idx, vividness_idx, GoF_idx, seq_len, do_timedir)


                    # update metadata
                    hc_count+=1
                    clarinet_count+=1
                    heard_count+=1

                    sub_X.append(sample)
                    #print("appended "+str(sample)+" to sub_X")
                    sub_y.append(label)
                    #print("appended "+str(label)+" to sub_y")

                # create Heard Trumpet training data
                for HT_train_idx in HT_train_idxs:
                    sample, label = get_samplelabel_single(HT_train_idx, sub_cycles, sub_allruns, CLS, run_idx, cycle_idx, cond_idx, timbre_idx, majorkey_idx, stimnote_idx, stimoctave_idx, vividness_idx, GoF_idx, seq_len, do_timedir)

                    # update metadata
                    ht_count+=1
                    trumpet_count+=1
                    heard_count+=1

                    sub_X.append(sample)
                    #print("appended "+str(sample)+" to sub_X")
                    sub_y.append(label)
                    #print("appended "+str(label)+" to sub_y")

                # create Heard Clarinet validation data
                for HC_val_idx in HC_val_idxs:
                    sample, label = get_samplelabel_single(HC_val_idx, sub_cycles, sub_allruns, CLS, run_idx, cycle_idx, cond_idx, timbre_idx, majorkey_idx, stimnote_idx, stimoctave_idx, vividness_idx, GoF_idx, seq_len, do_timedir)

                    # update metadata
                    val_hc_count+=1
                    val_clarinet_count+=1
                    val_heard_count+=1

                    sub_val_X.append(sample)
                    #print("appended "+str(sample)+" to sub_X")
                    sub_val_y.append(label)
                    #print("appended "+str(label)+" to sub_y")

                # create Heard Trumpet validation data
                for HT_val_idx in HT_val_idxs:
                    sample, label = get_samplelabel_single(HT_val_idx, sub_cycles, sub_allruns, CLS, run_idx, cycle_idx, cond_idx, timbre_idx, majorkey_idx, stimnote_idx, stimoctave_idx, vividness_idx, GoF_idx, seq_len, do_timedir)

                    # update metadata
                    val_ht_count+=1
                    val_trumpet_count+=1
                    val_heard_count+=1

                    sub_val_X.append(sample)
                    #print("appended "+str(sample)+" to sub_X")
                    sub_val_y.append(label)
                    #print("appended "+str(label)+" to sub_y")

                # do it all again if we're including imagined data
                # please for the love of god modularize all this
                if include_imagined:
                    # create Imagined Clarinet training data
                    for IC_train_idx in IC_train_idxs:
                        sample, label = get_samplelabel_single(IC_train_idx, sub_cycles, sub_allruns, CLS, run_idx,
                                                               cycle_idx, cond_idx, timbre_idx, majorkey_idx, stimnote_idx, stimoctave_idx, vividness_idx, GoF_idx, seq_len, do_timedir)
                        # update metadata
                        ic_count+=1
                        clarinet_count+=1
                        imagined_count+=1

                        sub_X.append(sample)
                        # print("appended "+str(sample)+" to sub_X")
                        sub_y.append(label)
                        # print("appended "+str(label)+" to sub_y")

                    # create Imagined Trumpet training data
                    for IT_train_idx in IT_train_idxs:
                        sample, label = get_samplelabel_single(IT_train_idx, sub_cycles, sub_allruns, CLS, run_idx,
                                                               cycle_idx, cond_idx, timbre_idx, majorkey_idx, stimnote_idx, stimoctave_idx, vividness_idx, GoF_idx, seq_len, do_timedir)
                        # update metadata
                        it_count+=1
                        trumpet_count+=1
                        imagined_count+=1

                        sub_X.append(sample)
                        # print("appended "+str(sample)+" to sub_X")
                        sub_y.append(label)
                        # print("appended "+str(label)+" to sub_y")

                    # create Imagined Clarinet validation data
                    for IC_val_idx in IC_val_idxs:
                        sample, label = get_samplelabel_single(IC_val_idx, sub_cycles, sub_allruns, CLS, run_idx,
                                                               cycle_idx, cond_idx, timbre_idx, majorkey_idx, stimnote_idx, stimoctave_idx, vividness_idx, GoF_idx, seq_len, do_timedir)
                        # update metadata
                        val_ic_count+=1
                        val_clarinet_count+=1
                        val_imagined_count+=1

                        sub_val_X.append(sample)
                        # print("appended "+str(sample)+" to sub_X")
                        sub_val_y.append(label)
                        print("appended "+str(label)+" to sub_y")

                    # create Imagined Trumpet validation data
                    for IT_val_idx in IT_val_idxs:
                        sample, label = get_samplelabel_single(IT_val_idx, sub_cycles, sub_allruns, CLS, run_idx,
                                                               cycle_idx, cond_idx, timbre_idx, majorkey_idx, stimnote_idx, stimoctave_idx, vividness_idx, GoF_idx, seq_len, do_timedir)
                        # update metadata
                        val_it_count+=1
                        val_trumpet_count+=1
                        val_imagined_count+=1

                        sub_val_X.append(sample)
                        # print("appended "+str(sample)+" to sub_X")
                        sub_val_y.append(label)
                        # print("appended "+str(label)+" to sub_y")

                # print all that crap for this subject
                metadata_dict = {"heard_count":heard_count, "hc_count":hc_count, "ht_count":ht_count, "val_heard_count":val_heard_count, "val_hc_count":val_hc_count, "val_ht_count":val_ht_count,
                                 "imagined_count":imagined_count, "ic_count":ic_count, "it_count":it_count, "val_imagined_count":val_imagined_count, "val_ic_count":val_ic_count, "val_it_count":val_it_count, "clarinet_count":clarinet_count, "trumpet_count":trumpet_count, "val_clarinet_count":val_clarinet_count, "val_trumpet_count":val_trumpet_count, "Length of train X":len(sub_X), "Length of train y":len(sub_y), "Length of val X":len(sub_val_X), "Length of val y":len(sub_val_y)
                                 }

                print("Metadata for subject "+str(sub)+": "+str(metadata_dict))

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
        if holdout_subjects == None:
            num_folds = 1
        else:
            # if we're holding out subjects or only runs from some amount of subjects
            # holout_subjects is defined up at the top of __main__
            hold_str = holdout_subjects.split(":") # something like 2:all
            held_per_fold = int(hold_str[0]) # number of subjects to be held out in each split
            runs_held = hold_str[1] # either half or all

            # this shuffled subject list is the same as was used for making NTP with heldout subs
            shuffled_subs = ['1668', '1410', '1678', '1088', '1672', '1419', '1571', '1541', '1401', '1660', '1664',
                                '1427', '1680', '1125', '1581', '1665', '1661']            #random.shuffle(shuffled_subs)
            num_folds = len(shuffled_subs)//held_per_fold # integer division, some subject may not be held out at all
            print("num folds is "+str(num_folds))
        for fold in range(0, num_folds):
            all_X = []
            all_y = []
            all_val_X = []
            all_val_y = []
            # load each subjects's contributions

            held_subs = []
            if holdout_subjects is not None:
                held_idx = fold*held_per_fold
                print("Held index for fold {0}: {1}".format(fold, held_idx))
                for k in range(0, held_per_fold):
                    held_subs.append(shuffled_subs[held_idx+k])
                print("For fold {0}, holding out {1}".format(fold, held_subs))


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
                # if held_subs is empty, we're not holding out subjects and the way it was saved is correct vis a vis training/validation
                if len(held_subs)==0:
                    all_X.extend(sub_X)
                    all_y.extend(sub_y)
                    all_val_X.extend(sub_val_X)
                    all_val_y.extend(sub_val_y)
                    print("extended "+str(shortsubid))
                else:
                    if sub in held_subs: # put stuff in validation
                        if runs_held == "all":
                            all_val_X.extend(sub_X)
                            all_val_y.extend(sub_y)
                            all_val_X.extend(sub_val_X)
                            all_val_y.extend(sub_val_y)
                            print("extended " + str(shortsubid))

                        elif runs_held == "half":
                            all_X.extend(sub_X)
                            all_y.extend(sub_y)
                            all_val_X.extend(sub_val_X)
                            all_val_y.extend(sub_val_y)
                            print("extended " + str(shortsubid))
                        else:
                            print("runs held was illegal value, got "+str(runs_held)+", quitting...")
                            quit(0)

                    else: # not held out, put everything in training
                        all_X.extend(sub_X)
                        all_y.extend(sub_y)
                        all_X.extend(sub_val_X)
                        all_y.extend(sub_val_y)
                        print("extended " + str(shortsubid))

            print("all_X has length"+str(len(all_X)))
            print("all_y has length"+str(len(all_y)))
            print("all_val_X has length"+str(len(all_val_X)))
            print("all_val_y has length"+str(len(all_val_y)))

            all_save_path = save_path + "all_"
            fold_str = ""
            if(num_folds > 1):
                print("Fold "+str(fold)+":")
                fold_str = "_fold{0}".format(fold)
            with open(all_save_path + "X"+fold_str+".p", "wb") as temp_fp:
                pickle.dump(all_X, temp_fp)
            with open(all_save_path + "y"+fold_str+".p", "wb") as temp_fp:
                pickle.dump(all_y, temp_fp)
            with open(all_save_path + "val_X"+fold_str+".p", "wb") as temp_fp:
                pickle.dump(all_val_X, temp_fp)
            with open(all_save_path + "val_y"+fold_str+".p", "wb") as temp_fp:
                pickle.dump(all_val_y, temp_fp)
            print("forward count is "+str(forward_count))
            print("reverse count is "+str(reverse_count))
            print("Saved concatenated training data for all subjects")





