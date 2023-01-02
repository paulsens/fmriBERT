import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import itertools
import os
import pickle
import pandas as pd
from Constants import *
from random import randint
from datetime import date
from torch.utils.data import Dataset
from helpers import standardize_flattened, detrend_flattened
import random
import copy as coppy


random_numbers=[]
# make the pretraining datasets according to several parameters
#  the first is the probability threshold for inclusion in STG as a string, e.g "23", the second is "left" or "right"
#   the third is the length of each half of the sample either 5 or 10, num_copies is the number of positive and negative training samples to create from each left-hand reference sample, test_copies is the number of repetitions we want from the -test- runs, from 1 to 4. Recall that each test run is the same 10 clips repeated four times. Default is 1, i.e each set of 10 only once/only the first ten from each test run.
#     The second to last two determine the CLS and MSK tasks that will be trained on.
# the default allowed genres is all of them, 0 to 9 inclusive
def make_ptdsingle_data(threshold, hemisphere,  allowed_genres, seq_len=10, num_copies=1, include_test=1, test_copies=1, val_copies=1, standardize=1, detrend="linear", within_subjects=1, binary="arrowoftime", mask_task="reconstruct", seed=3, val_flag=0, verbose=1):
    #runs_dict is defined in Constants.py
    test_runs = runs_dict["Test"]
    training_runs = runs_dict["Training"]
 # for testing that needs reproducibility

    #size of left or right STG voxel space, with 3 token dimensions already added in
    voxel_dim = COOL_DIVIDEND+NUM_TOKENS #defined in Constants.py
    CLS = [1] + ([0] * (voxel_dim - 1))  # first dimension is reserved for cls_token flag
    MSK = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
    # count the total number of samples for each direction, should be roughly the same
    reverse_count = 0
    forward_count = 0
    if val_flag:
        holdout_range=range(0,12)
    else:
        holdout_range=range(0,1)

    random.seed(seed) #seed is passed in as a parameter, defaults to 3
    # making k-many datasets for k-fold crossvalidation where k=len(holdout_range)
    # i.e each fold has one run held out
    for holdout in holdout_range:

        training_samples = []
        # final list of labels for training
        training_labels = []
        # keep count of how many sample/label pairs we've created
        count = 0
        val_samples = [] # samples drawn from the heldout run
        val_labels = []
        val_count = 0

        # the reference list of left-hand samples, index set by the count variable
        #  find its genre with ref_to_genre_dict
        ref_samples = []
        # reference list of genres, where element i is the genre of ref_samples[i]
        ref_genres = []

        # give the sub id as a key to this dictionary to obtain the genre_sample_dict dictionary for that subject
        #   that dictionary is defined below inside the loop
        sub_genre_sample_dict = {}
        sub_startstop_list = []
        ref_to_genre_dict = {}

        # which of the training runs is being held out to make this file
        # this number will be passed as a command line arg to pretrain.py to know which data to load
        for sub in ["001", "002", "003", "004", "005"]:
            # a list of indices for each genre, where to find each genre in the aggregate training data list
            #  give sub as a key to sub_genre_sample_dict to obtain the dictionary below
            genre_sample_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
            iter=0 #iterations of the next loop, resets per subject

            #opengenre_preproc_path is defined in Constants.py
            subdir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub + "/"
            #load voxel data and labels
            with open(subdir+"STG_allruns"+hemisphere+"_t"+threshold+"_"+str(NUM_TOKENS)+"tokens.p", "rb") as data_fp:
                all_data = pickle.load(data_fp)
            with open(subdir+"labelindices_allruns.p","rb") as label_fp:
                # i made these labels back in april but i don't see any reason why they would have changed
                # also i dont think i even need them for unsupervised pretraining
                all_labels = pickle.load(label_fp)
                # THE TSV FILES ON THE DATASET WEBSITE ASSIGN GENRES TO THE DUMMY 15s AT THE BEGINNING OF EACH RUN
                # THESE ARE NOT REAL AND ARE NOT INCLUDED IN THE LOADED FILES
            voxel_data=[]
            labels=[]


            # detrend contains the name of the detrending we want, e.g linear or spline, within subjects is a binary flag
            # function is defined in helpers.py
            if(detrend!=None and within_subjects):
                all_data=detrend_flattened(all_data, detrend, num_tokens=NUM_TOKENS)
            # if the standardize flag is set, set mean to zero and variance to 1
            # function is defined in helpers.py
            if(standardize):
                all_data = standardize_flattened(all_data, num_tokens=NUM_TOKENS)
            # only remove the test_copies many repetitions (max 4) during the Test Runs
            # note that the dummy data described in the dataset documentation has already been removed, so we are start at index 0 in the loaded data
            n_test_runs = runs_dict["Test"]
            amount = OPENGENRE_TRSPERRUN*test_copies//4
            for run in range(0, n_test_runs):
                start = run*OPENGENRE_TRSPERRUN
                for TR in range(start, start+amount):
                    voxel_data.append(all_data[TR])
                    if(TR%10==0): # each label corresponds to 10 TRs
                        labels.append(all_labels[TR//10]) # index in the labels list is one tenth floor'd of the data index
            #add every TR from the remaining 12 Training Runs, no repetitions here
            start = n_test_runs*OPENGENRE_TRSPERRUN
            holdout_start = len(voxel_data) +(OPENGENRE_TRSPERRUN*holdout)#record where the heldout run begins
            holdout_stop = holdout_start + OPENGENRE_TRSPERRUN # and where to stop
            #but these values refer to the voxel_data list, which hasn't been made into sequences yet
            holdout_start = holdout_start/seq_len #account for when the seq_len sequences get made
            holdout_stop = holdout_stop/seq_len

            #print("When holding out run "+str(holdout), "start and stop are "+str(holdout_start)+","+str(holdout_stop))
            for TR in range(start, len(all_data)):
                voxel_data.append(all_data[TR])
                if (TR % 10 == 0):  # each label corresponds to 10 TRs
                    #print("When TR is "+str(TR))
                    labels.append(all_labels[TR // 10])  # index in the labels list is one tenth floor'd of the data index

            #labels are 0 through 9 inclusive
            #print("length of voxel data after applying test_copies "+str(len(voxel_data)))
            #print("length of labels after applying test_copies is "+str(len(labels)))


            timesteps = len(voxel_data)

            # THIS BLOCK IS CREATING REF_SAMPLES, WHICH HAS ALL SUBJECTS AND ALL RUNS
            # THE HELDOUT RUN IS ALSO HERE, BUT WE KEEP TRACK OF WHERE IT IS
            #we're going to obtain timesteps//seq_len many samples from each subject
            for t in range(0, timesteps//seq_len):
                start = t*seq_len
                this_sample = [] #create the left-hand sample starting at TR number {start}
                # add the seq_len many next images
                for s in range(0, seq_len):
                    this_sample.append(voxel_data[start+s])

                this_genre = labels[iter//(10//seq_len)] #the divisor is either 2 or 1

                #add count as an index corresponding to this genre
                genre_sample_dict[this_genre].append(count)
                ref_to_genre_dict[count]=this_genre
                #append to aggregate lists
                ref_samples.append(this_sample)
                ref_genres.append(this_genre)
                #now ref_samples[i] is a left hand sample with genre ref_genres[i]

                #increase count
                count = count+1
                iter = iter+1
            #this subject's genre_sample_dict is done, save it in the parent dictionary with sub as key
            sub_genre_sample_dict[sub]=genre_sample_dict
            print("after subject "+str(sub)+", ref samples has length "+str(len(ref_samples)))

        #iter counts how many left samples we make per subject, should be the same for each
        # so after all the subjects are done this value should linger from the final subject's for-loop
        samples_per_subject=iter
        print("samples per subject is "+str(samples_per_subject))

        # THIS BLOCK STORES THE INDICES INTO REF_SAMPLES FOR EACH SUBJECT'S HELDOUT DATA
        for m in range(0,5): #magic number for number of subjects num_subs doesn't exist yet
            # holdout_start indexes into each subject's chunk within ref_samples
            # so multiply up to m's chunk and then get the index where the heldout run begins and ends for them
            sub_startstop_list.append([holdout_start+(samples_per_subject*m), holdout_stop+(samples_per_subject*m)])
        print("sub startstop list for holdout "+str(holdout)+" is "+str(sub_startstop_list)+"\n")
        # reference lists are done with threshold, hemisphere, seq_len, detrending, standardization, and test_copies applied
        # the following code builds the training samples and labels by applying num_copies, allowed_genres, and creating both a positive and negative sample for the CLS token task

        # BUILD THE SET OF INPUTS AND LABELS HOLDING OUT THE CURRENT HOLDOUT RUN
        for i in range(0, len(ref_samples)):

            sub_id_int = (i//samples_per_subject) + 1 #e.g 1100//500 = 2, and subject 3 has range 1000 to 1499 if samplespersubject is 500
            sub_start = (sub_id_int - 1) * samples_per_subject
            sub_end = sub_start + samples_per_subject - 1 #the last index that is included for that subject
            if (verbose and i % 1000 == 0):
                print("Sub start is {start}, and sub end is {end}".format(start=str(sub_start), end=str(sub_end)))
            startstop_list = sub_startstop_list[sub_id_int-1] #get holdout indices
            this_holdout_start=startstop_list[0]
            this_holdout_stop = startstop_list[1]
            if i < this_holdout_stop and i >= this_holdout_start:
                heldout = True
                #heldout = False #for testing purposes to not hold stuff out
            else:
                heldout = False
            sub_id = "00"+str(sub_id_int) # turn it into the familiar id string
            #print("sub_id is "+str(sub_id))

            this_genre = ref_genres[i] # genre of left-hand sample
            if this_genre not in allowed_genres:
                continue #skip the rest of this iteration

            if(binary=="arrowoftime"): # arrow of time task, possibly called temporal orientation instead?
                # 50-50 chance of this sequence being reversed, decode whether it's reversed (1) or not (0)
                # the question at hand, then, is "HAS THIS SEQUENCE BEEN REVERSED?"
                this_input=[coppy.deepcopy(CLS)]
                # 1 means reversed, 0 means forward
                direction = random.randint(0,1)
                #print("direction is "+str(direction))

                # this if-else block sets up the reversal or non-reversal
                if direction==0:
                    # has not been reversed
                    forward_count+=1
                    start_idx = 0
                    end_idx = seq_len
                    incr = 1
                else:
                    # has been reversed, direction==1
                    #print("we're reversing")
                    reverse_count+=1
                    start_idx = seq_len-1
                    end_idx = -1
                    incr = -1

                for j in range(start_idx, end_idx, incr):
                    this_input.append(coppy.deepcopy(ref_samples[i][j]))  # fill left hand sample

                if direction==1 and (reverse_count==0 or reverse_count==1):
                    print("Original sequence was "+str(ref_samples[i]))
                    print("And this input is "+str(this_input))

                # i use an augmented label to smuggle in some metadata that I might want
                # can't include strings in order to be converted a tensor properly during training
                this_label = [direction, this_genre, i, sub_id_int]
                # add sample and label to final product

                if heldout:
                    val_samples.append(this_input)  # append to heldout validation set
                    val_labels.append(this_label)
                else:
                    training_samples.append(this_input)
                    training_labels.append(this_label)

        if(verbose):
            print("Training_samples has length "+str(len(training_samples))+", and each element has length "+str(len(training_samples[0]))+". Each of those has length "+str(len(training_samples[0][0]))+".\n\n")
            print("Val_samples has length "+str(len(val_samples))+", and each element has length "+str(len(val_samples[0]))+". Each of those has length "+str(len(val_samples[0][0]))+".\n\n")
            #print("Val labels has length" +str(len(val_labels)))

        #save training_samples and training_labels
        time = date.today()
        #this_dir = opengenre_preproc_path+"training_data/cross_val/"+str(time)+"/"
        this_dir = opengenre_preproc_path+"training_data/timedir/"+str(time)+"/"

        if not os.path.exists(this_dir):
            os.mkdir(this_dir)

        # save the refsamples list, this should be the same every time so no need to keep track with a count in the filename
        with open(this_dir+str(hemisphere)+"_refsamples.p","wb") as refsamples_fp:
            pickle.dump(ref_samples, refsamples_fp)
        count = 0
        # set the last part of the filename by checking what already exists
        this_file = this_dir + str(hemisphere) + "_samples" + str(count) + ".p"
        while os.path.exists(this_file):
            count += 1
            this_file = this_dir + str(hemisphere) + "_samples" + str(count) + ".p"


        #save training data and labels
        with open(this_file,"wb") as samples_fp:
            pickle.dump(training_samples,samples_fp)
        with open(this_dir+str(hemisphere)+"_labels"+str(count)+".p","wb") as labels_fp:
            pickle.dump(training_labels,labels_fp)

        #save the test data and labels
        with open(this_dir+str(hemisphere)+"_valsamples"+str(count)+".p","wb") as valsamples_fp:
            pickle.dump(val_samples, valsamples_fp)
        with open(this_dir+str(hemisphere)+"_vallabels"+str(count)+".p","wb") as vallabels_fp:
            pickle.dump(val_labels,vallabels_fp)

        #save metadata
        with open(this_dir+str(hemisphere)+"_metadata"+str(count)+".txt","w") as meta_fp:
            meta_fp.write("Making timedirection binary task data with the 12 heldout run crossval scheme. Seeding rng with seed=3 at the beginning.\n"+
                    "\nnum_samples:"+str(len(training_samples))+
                    "\nallowed_genres:"+str(allowed_genres)+
                    "\nthreshold:"+str(threshold)+
                    "\nhemisphere:"+str(hemisphere)+
                    "\nseq_len:"+str(seq_len)+
                    "\nnum_copies:"+str(num_copies)+
                    "\ntest_copies:"+str(test_copies)+
                    "\nbinary:"+str(binary)+
                    "\nmask_task:"+str(mask_task)+
                    "\ncount:"+str(count)+
                    "\nvoxel_dim:"+str(voxel_dim)+
                    "\nrandom seed: "+str(seed)+
                    "\nRandom numbers were: "+str(random_numbers)+
                          "\n")
    print("Final reverse count: "+str(reverse_count))
    print("Final forward count: "+str(forward_count))
if __name__=="__main__":
    #seq_len should be either 5 or 10
    #hemisphere is left or right
    #threshold only 23 right now
    #num_copies is the number of positive and negative training samples to create from each left-hand sample
    #allowed_genres is a list of what it sounds like, remember range doesn't include right boundary
    # val_copies is the data augmentation factor for the held out run. That run will have ~1/10 the data of the full run, so it probably needs some augmentation
    make_ptdsingle_data("23", hemisphere="left", seq_len=10, num_copies=1, standardize=1, detrend="linear", mask_task="same_genre", within_subjects=1, allowed_genres=range(0,10), val_flag=1)