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

###### FUNCTIONS #######
WINDOW_LENGTH = 5
def make_ntp_data(sub_allruns, run_list=None, TRSperrun=223, startTR=None, stride=None, pos_and_neg=False):
    X_sub = []
    y_sub = []
    pos_count = 0
    neg_count = 0

    for run_n in run_list:  # list of ints
        window_start = (TRSperrun * run_n) + startTR
        window_end = window_start + WINDOW_LENGTH # window_end is one greater than the last included TR
        end_TR = TRSperrun*(run_n+1)

        # print("window start: " + str(window_start))
        # print("window end: " + str(window_end))

        counter = 0
        while (window_end+WINDOW_LENGTH)<=end_TR:

            this_x = [copy.deepcopy(CLS)]
            for i in range(0, WINDOW_LENGTH):
                this_x.append(copy.deepcopy(sub_allruns[window_start+i]))
            this_x.append(copy.deepcopy(SEP))

            ###### make a positive sample by grabbing the next WINDOW_LENGTH-many TRs
            x_pos = copy.deepcopy(this_x)
            y_pos = [1]
            #print("for x_neg, appending TRs "+str(window_end)+" through "+str(window_end+WINDOW_LENGTH))

            for i in range(0, WINDOW_LENGTH):
                if (window_end + i) > end_TR:
                    print("something has gone wrong, window end + i is "+str(window_end + i)+" and end_TR is "+str(end_TR)+", quitting...")
                    quit(0)
                x_pos.append(copy.deepcopy(sub_allruns[window_end+i]))

            ###### make a negative sample with a buffer zone around the current window
            x_neg = copy.deepcopy(this_x)
            y_neg = [0]

            buffer_TRs = list(range(window_start - WINDOW_LENGTH, window_start + WINDOW_LENGTH*3))
            negative_start = window_start # guaranteed failure to initialize the loop
            possible_starts = list(range(3, len(sub_allruns)-WINDOW_LENGTH)) # the negative sample can come from any run
            while negative_start in buffer_TRs:
                negative_start = random.choice(possible_starts)

            #print("for x_neg, appending TRs "+str(negative_start)+" through "+str(negative_start+WINDOW_LENGTH))
            for i in range(0, WINDOW_LENGTH):
                x_neg.append(copy.deepcopy(sub_allruns[negative_start+i]))

            #print("after construction, x_pos has length "+str(len(x_pos))+" and x_neg has length "+str(len(x_neg)))
            if pos_and_neg:
                X_sub.append(x_pos)
                y_sub.append(y_pos)
                pos_count+=1
                X_sub.append(x_neg)
                y_sub.append(y_neg)
                neg_count+=1


            else:
                choice = random.choice([0,1])
                if choice==0: # 0 means false, it is NOT the true next thought
                    X_sub.append(x_neg)
                    y_sub.append(y_neg)
                    neg_count+=1

                else: # 1 means true, it IS the true next thought
                    X_sub.append(x_pos)
                    y_sub.append(y_pos)
                    pos_count+=1


            ###### increment window position
            window_start += stride
            window_end += stride
            counter+=1

    print("before return, pos_count is "+str(pos_count)+" and neg_count is "+str(neg_count))

    return X_sub, y_sub


def make_sametimbre_data(sub_allruns, run_list=None, TRSperrun=223, pos_and_neg=False):
    cyclesperrun = 21

    # TRs are zero indexed, first stimulus is on TR 8 (line 9 in the text file)
    # HRF peaks after 6.5 seconds or 3 TRs, so it probably makes the most sense to do STIM STIM REST PROBE REST since the probe is always the same timbre
    startTR = 8
    TRs_between_cycles = 11


    X_sub = []
    y_sub = []
    pos_count = 0
    neg_count = 0

    for run_n in run_list:  # list of ints
        window_start = (TRSperrun * run_n) + startTR
        window_end = window_start + WINDOW_LENGTH # window_end is one greater than the last included TR
        end_TR = TRSperrun*(run_n+1)

        # print("window start: " + str(window_start))
        # print("window end: " + str(window_end))

        counter = 0
        while (window_end+WINDOW_LENGTH)<=end_TR:

            this_x = [copy.deepcopy(CLS)]
            for i in range(0, WINDOW_LENGTH):
                this_x.append(copy.deepcopy(sub_allruns[window_start+i]))
            this_x.append(copy.deepcopy(SEP))

            ###### make a positive sample by grabbing the next WINDOW_LENGTH-many TRs
            x_pos = copy.deepcopy(this_x)
            y_pos = [1]
            #print("for x_neg, appending TRs "+str(window_end)+" through "+str(window_end+WINDOW_LENGTH))

            for i in range(0, WINDOW_LENGTH):
                if (window_end + i) > end_TR:
                    print("something has gone wrong, window end + i is "+str(window_end + i)+" and end_TR is "+str(end_TR)+", quitting...")
                    quit(0)
                x_pos.append(copy.deepcopy(sub_allruns[window_end+i]))

            ###### make a negative sample with a buffer zone around the current window
            x_neg = copy.deepcopy(this_x)
            y_neg = [0]

            buffer_TRs = list(range(window_start - WINDOW_LENGTH, window_start + WINDOW_LENGTH*3))
            negative_start = window_start # guaranteed failure to initialize the loop
            possible_starts = list(range(3, len(sub_allruns)-WINDOW_LENGTH)) # the negative sample can come from any run
            while negative_start in buffer_TRs:
                negative_start = random.choice(possible_starts)

            #print("for x_neg, appending TRs "+str(negative_start)+" through "+str(negative_start+WINDOW_LENGTH))
            for i in range(0, WINDOW_LENGTH):
                x_neg.append(copy.deepcopy(sub_allruns[negative_start+i]))

            #print("after construction, x_pos has length "+str(len(x_pos))+" and x_neg has length "+str(len(x_neg)))
            if pos_and_neg:
                X_sub.append(x_pos)
                y_sub.append(y_pos)
                pos_count+=1
                X_sub.append(x_neg)
                y_sub.append(y_neg)
                neg_count+=1


            else:
                choice = random.choice([0,1])
                if choice==0: # 0 means false, it is NOT the true next thought
                    X_sub.append(x_neg)
                    y_sub.append(y_neg)
                    neg_count+=1

                else: # 1 means true, it IS the true next thought
                    X_sub.append(x_pos)
                    y_sub.append(y_pos)
                    pos_count+=1


            ###### increment window position
            window_start += stride
            window_end += stride
            counter+=1

    print("before return, pos_count is "+str(pos_count)+" and neg_count is "+str(neg_count))

    return X_sub, y_sub
###### parameters and paths
NUM_TOKENS = 2  # number of dimensions reserved by tokens, e.g CLS/MSK
hemisphere="right"
threshold="23"
ROI="STG"
n_runs=8 #all subjects had 8 runs in this dataset
sublist = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665", "1668", "1672", "1678", "1680"]
sublist_shuffled = ['1668', '1410', '1678', '1088', '1672', '1419', '1571', '1541', '1401', '1660', '1664', '1427', '1680', '1125', '1581', '1665', '1661']

if NUM_TOKENS == 2:
    ROIfile_str=ROI+"_allruns"+str(hemisphere)+"_t"+str(threshold)+"_"+str(NUM_TOKENS)+"tokens.p"
else:
    ROIfile_str=ROI+"_allruns"+str(hemisphere)+"_t"+str(threshold)+".p"#where is the original data loaded from?
data_path = pitchclass_preproc_path
# save training_samples and training_labels
time = date.today()
seq_len = 5
forward_count=0
reverse_count=0
voxel_dim = 420

CLS = [1] + ([0] * (voxel_dim - 1))  # first dimension is reserved for cls_token flag
MSK = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
SEP = [0, 0, 1] + ([0] * (voxel_dim - 3))  # third dimension is reserved for sep_token flag

Xtrain_allsubs = []
ytrain_allsubs = []
Xval_allsubs = []
yval_allsubs = []

stride = 2 # how much does the window move after constructing a sample? set to WINDOW_LENGTH (defined at top) if you don't want overlap
p_and_n = True # include positive and negative sample for each 5seq or not

if __name__=="__main__":
    # task is either sametimbre or ntp
    #task = "sametimbre"
    task = "sametimbre"
    fold = 0
    holdout_subjects = 2  # number of subjects to hold out per fold. 0 means that runs are held out instead

    if task == "ntp":
        if holdout_subjects == 0:
            save_path = "/Volumes/External/pitchclass/pretraining/" + str(time) + "-" + str(seq_len) + "TR_" + str(
                stride) + "stride_" + str(p_and_n) + "posneg/"
        else:
            save_path = "/Volumes/External/pitchclass/pretraining/" + str(time) + "-" + str(seq_len) + "TR_" + str(
                stride) + "stride_" + str(p_and_n) + "posneg_" + str(holdout_subjects) + "heldout/"
    elif task == "sametimbre":
        if holdout_subjects == 0:
            save_path = "/Volumes/External/pitchclass/finetuning/sametimbre/datasets/" + str(time) + "-" + str(seq_len) + "TR_" + "hasimagined_" + str(p_and_n) + "posneg/"
        else:
            save_path = "/Volumes/External/pitchclass/finetuning/sametimbre/datasets/" + str(time) + "-" + str(seq_len) + "TR_" + "hasimagined_" + str(p_and_n) + "posneg_" + str(holdout_subjects) + "heldout_"+str(hemisphere)+"STG/"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # grab the heldout IDs from the shuffled list
    heldout_subIDs = []
    if holdout_subjects>0:
        for i in range(0, holdout_subjects):
            heldout_subIDs.append(sublist_shuffled[fold*holdout_subjects+i])

    for sub in sublist:
        # load flattened data
        subid = "sub-sid00" + sub
        shortsubid = "sid00" + sub
        full_path = data_path + subid + "/" + subid + "/" + ROIfile_str
        with open(full_path, "rb") as allruns_fp:
            sub_allruns = pickle.load(allruns_fp)
        xlen = len(sub_allruns)
        ylen = len(sub_allruns[0])

        # detrend and standardize
        sub_allruns = detrend_flattened(sub_allruns, detrend="linear", num_tokens=NUM_TOKENS)
        sub_allruns = standardize_flattened(sub_allruns, num_tokens=NUM_TOKENS)


        TRSperrun = 233
        startTR = 3 # 0 and 1 are dummy TRs, start on 3 to align with the final TR

        window_start = startTR
        window_end = window_start + stride

        if task=="ntp":
            if holdout_subjects == 0:
                X_training, y_training = make_ntp_data(sub_allruns, run_list = [0, 1, 2, 3], TRSperrun = TRSperrun, startTR = startTR, stride=stride, pos_and_neg=p_and_n)

                X_val, y_val = make_ntp_data(sub_allruns, run_list = [4, 5, 6, 7], TRSperrun = TRSperrun, startTR = startTR, stride=stride, pos_and_neg=p_and_n)

            else: # if we're holding out subjects, check if this one is held out
                if sub in heldout_subIDs:
                    X_training = [] # this sub contributes nothing to training data
                    y_training = []
                    X_val, y_val = make_ntp_data(sub_allruns, run_list = [0, 1, 2, 3, 4, 5, 6, 7], TRSperrun = TRSperrun, startTR = startTR, stride = stride, pos_and_neg=p_and_n)
                else: # if it's not held out, contribute everything to training data and nothing to val
                    X_training, y_training = make_ntp_data(sub_allruns, run_list = [0, 1, 2, 3, 4, 5, 6, 7], TRSperrun = TRSperrun, startTR = startTR, stride = stride, pos_and_neg=p_and_n)
                    X_val = []
                    y_val = []


        elif task=="sametimbre":
            if holdout_subjects == 0:
                X_training, y_training = make_sametimbre_data(sub_allruns, run_list = [0, 1, 2, 3], TRSperrun = TRSperrun, pos_and_neg=p_and_n)

                X_val, y_val = make_sametimbre_data(sub_allruns, run_list = [4, 5, 6, 7], TRSperrun = TRSperrun, pos_and_neg=p_and_n)

            else: # if we're holding out subjects, check if this one is held out
                if sub in heldout_subIDs:
                    X_training = [] # this sub contributes nothing to training data
                    y_training = []
                    X_val, y_val = make_sametimbre_data(sub_allruns, run_list = [0, 1, 2, 3, 4, 5, 6, 7], TRSperrun = TRSperrun, pos_and_neg=p_and_n)
                else: # if it's not held out, contribute everything to training data and nothing to val
                    X_training, y_training = make_sametimbre_data(sub_allruns, run_list = [0, 1, 2, 3, 4, 5, 6, 7], TRSperrun = TRSperrun,  pos_and_neg=p_and_n)
                    X_val = []
                    y_val = []

        Xtrain_allsubs.extend(X_training)
        ytrain_allsubs.extend(y_training)
        Xval_allsubs.extend(X_val)
        yval_allsubs.extend(y_val)

        print("After sub "+str(sub)+", Xtrain has length {0}, ytrain has length {1}, Xval has length {2}, and yval has length {3}".format(len(Xtrain_allsubs), len(ytrain_allsubs), len(Xval_allsubs), len(yval_allsubs)))

    all_save_path = save_path + "all_"

    if holdout_subjects == 0:
        with open(all_save_path + "X.p", "wb") as temp_fp:
            pickle.dump(Xtrain_allsubs, temp_fp)
        with open(all_save_path + "y.p", "wb") as temp_fp:
            pickle.dump(ytrain_allsubs, temp_fp)
        with open(all_save_path + "val_X.p", "wb") as temp_fp:
            pickle.dump(Xval_allsubs, temp_fp)
        with open(all_save_path + "val_y.p", "wb") as temp_fp:
            pickle.dump(yval_allsubs, temp_fp)

    else:
        with open(all_save_path + "X_fold"+str(fold)+".p", "wb") as temp_fp:
            pickle.dump(Xtrain_allsubs, temp_fp)
        with open(all_save_path + "y_fold"+str(fold)+".p", "wb") as temp_fp:
            pickle.dump(ytrain_allsubs, temp_fp)
        with open(all_save_path + "val_X_fold"+str(fold)+".p", "wb") as temp_fp:
            pickle.dump(Xval_allsubs, temp_fp)
        with open(all_save_path + "val_y_fold"+str(fold)+".p", "wb") as temp_fp:
            pickle.dump(yval_allsubs, temp_fp)









