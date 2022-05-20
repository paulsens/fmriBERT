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



# make the pretraining datasets according to several parameters
#  the first is the probability threshold for inclusion in STG as a string, e.g "23", the second is "left" or "right"
#   the third is the length of each half of the sample either 5 or 10, num_copies is the number of positive and negative training samples to create from each left-hand reference sample, test_copies is the number of repetitions we want from the -test- runs, from 1 to 4. Recall that each test run is the same 10 clips repeated four times. Default is 1, i.e each set of 10 only once/only the first ten from each test run.
#     The second to last two determine the CLS and MSK tasks that will be trained on.
# the default allowed genres is all of them, 0 to 9 inclusive
def make_pretraining_data(threshold, hemisphere,  allowed_genres, seq_len=5, num_copies=1, test_copies=1,standardize=1, detrend="linear", within_subjects=1, binary="same genre", multiclass="genre_decode", verbose=1, test_split=None):
    #runs_dict is defined in Constants.py
    test_runs = runs_dict["Test"]
    training_runs = runs_dict["Training"]
    random.seed(3)  # for testing that needs reproducibility

    #size of left or right STG voxel space, with 3 token dimensions already added in
    voxel_dim = COOL_DIVIDEND+3 #defined in Constants.py
    CLS = [1] + ([0] * (voxel_dim - 1))  # first dimension is reserved for cls_token flag
    MSK = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
    SEP = [0, 0, 1] + ([0] * (voxel_dim - 3))  # third dimension is reserved for sep_token flag

    #each element of this list should be (seq_len*2 + 2,real_voxel_dim+3), i.e CLS+n TRs+SEP+n TRs
    #   where 3 extra dimensions have been added to the front of the voxel space for the tokens
    training_samples = []
    # final list of labels for training
    training_labels = []
    #keep count of how many sample/label pairs we've created
    count = 0

    # the reference list of left-hand samples, index set by the count variable
    #  find its genre with ref_to_genre_dict
    ref_samples = []
    # reference list of genres, where element i is the genre of ref_samples[i]
    ref_genres = []

    # give the sub id as a key to this dictionary to obtain the genre_sample_dict dictionary for that subject
    #   that dictionary is defined below inside the loop
    sub_genre_sample_dict = {}
    ref_to_genre_dict = {}

    #loop through subjects
    for sub in ["001", "002", "003", "004", "005"]:
        # a list of indices for each genre, where to find each genre in the aggregate training data list
        #  give sub as a key to sub_genre_sample_dict to obtain the dictionary below
        genre_sample_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        iter=0 #iterations of the next loop, resets per subject
        #opengenre_preproc_path is defined in Constants.py
        subdir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub + "/"
        #load voxel data and labels
        with open(subdir+"STG_allruns"+hemisphere+"_t"+threshold+".p", "rb") as data_fp:
            all_data = pickle.load(data_fp)
        with open(subdir+"labelindices_allruns.p","rb") as label_fp:
            all_labels = pickle.load(label_fp)
        voxel_data=[]
        labels=[]


        #detrend contains the name of the detrending we want, e.g linear or spline, within subjects is a binary flag
        # function is defined in helpers.py
        if(detrend!=None and within_subjects):
            all_data=detrend_flattened(all_data, detrend)
        #if the standardize flag is set, set mean to zero and variance to 1
        #function is defined in helpers.py
        if(standardize):
            all_data = standardize_flattened(all_data)
        #only remove the test_copies many repetitions (max 4) during the Test Runs
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
        for TR in range(start, len(all_data)):
            voxel_data.append(all_data[TR])
            if (TR % 10 == 0):  # each label corresponds to 10 TRs
                labels.append(all_labels[TR // 10])  # index in the labels list is one tenth floor'd of the data index
        #labels are 0 through 9 inclusive
        print("length of voxel data after applying test_copies "+str(len(voxel_data)))
        print("length of labels after applying test_copies is "+str(len(labels)))



        timesteps = len(voxel_data)
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


    #iter counts how many left samples we make per subject, should be the same for each
    # so after all the subjects are done this value should linger from the final subject's for-loop
    samples_per_subject=iter


    # reference lists are done with threshold, hemisphere, seq_len, detrending, standardization, and test_copies applied
    # the following code builds the training samples and labels by applying num_copies, allowed_genres, and creating both a positive and negative sample for the CLS token task

    #test_split is given as a parameter to make_pretraining_data is an integer, ie 10 means 10%, so divide by 100
    n_test = int((test_split/100)*len(ref_samples))
    if(n_test != 0):
        test_idxs = random.sample(range(0,len(ref_samples)), n_test)
        #give each subject their own list of test_data and test labels to preserve within_subject functionality
        test_data_dict = {"001":[],"002":[],"003":[],"004":[],"005":[]}
        test_label_dict = {"001":[],"002":[],"003":[],"004":[],"005":[]}
        test_genre_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        tdd = test_data_dict #just a shorthand
        test_count=0
    else:
        test_idxs = None
        test_data_dict = None
        test_label_dict = None
        test_genre_count = 0
        tdd = None
        test_count = 0

    # for each left-hand sample, create num_copies positive and negative training samples
    for i in range(0, len(ref_samples)):
        sub_id = (i//samples_per_subject) + 1 #e.g 1100//500 = 2, and subject 3 has range 1000 to 1499 if samplespersubject is 500
        sub_id = "00"+str(sub_id) # turn it into the familiar id string
        #print("sub_id is "+str(sub_id))

        pos_partners = [] # the reference indices with same genre that have already been paired on the right hand side
        neg_partners = [] # the reference indices with different genre that have already been paired on the rhs
        lh_genre = ref_genres[i] # genre of left-hand sample
        if lh_genre not in allowed_genres:
            continue #skip the rest of this iteration
        if n_test != 0:
            if i in test_idxs:
                #make a new list for the test data, might be able to just use ref_samples[i] but being safe for now
                # with a new list
                test_left=[]
                test_label = ref_genres[i]
                test_genre_count[int(test_label)]+=1 # add one to the counter for that genre
                for k in range(0, seq_len):
                    test_left.append(ref_samples[i][k])
                test_data_dict[sub_id].append(test_left)
                test_label_dict[sub_id].append(test_label)
                test_count+=1 # increment count of test leftsamples
                #dont do the stuff below if this index was assigned to be test data
                continue
        for copy in range(0, num_copies):
            input_pos = [CLS] #create positive sample, get a new address for each iteration
            input_neg = [CLS] #create negative sample, get a new address for each iteration

            # create positive training sample, so same genre
            rh_genre = lh_genre
            for j in range(0, seq_len):
                input_pos.append(ref_samples[i][j]) # fill left hand sample
            input_pos.append(SEP) #add separator token
            partner = i #initial condition for loop
            while partner==i: #find a new partner with the same genre
                if (within_subjects): #if within_subjects flag is set
                    dict_key=sub_id #only look at the samples from this subject with this genre
                else: #pick one of the subjects at random to pair it with, including themself
                    dict_key = randint(1,5)
                    dict_key = "00"+str(dict_key)

                partner = random.choice(sub_genre_sample_dict[dict_key][lh_genre]) #all other reference indices of this genre for whichever subject was chosen
                # this seems roundabout, but if within_subjects is not set this is functionally the same as picking one at random from a complete list, while also allowing mostly re-used code if within_subjects is set
                #basically, i did it this way to allow the most overlap in code whether within_subjects is set or not

                if partner in pos_partners: #did we put this on the right side of this left sample already?
                    partner=i #if so, keep the loop going
                elif n_test != 0:
                    if partner in test_idxs:
                        partner=i #continue the loop if this index is assigned to test_data, we dont want
                               # test data indices on the left OR right side of training samples
                else:
                    pos_partners.append(partner) #otherwise put it in the list and we'll exit the loop
            for j in range(0, seq_len):
                input_pos.append(ref_samples[partner][j]) #fill partner as right hand sample
            pos_label = [1, lh_genre, rh_genre] #the labels for training, the 1 means same genre
            training_samples.append(input_pos) #add this input to the final list of training inputs
            training_labels.append(pos_label) #add corresponding positive label vector

            # create negative training sample, so get a different genre
            rh_genre = lh_genre #initial condition for the loop
            while rh_genre == lh_genre: #until we get a different one
                rh_genre = random.choice(allowed_genres) #get a genre label
            for j in range(0, seq_len):
                input_neg.append(ref_samples[i][j]) # fill left hand sample
            input_neg.append(SEP) #add separator token

            partner = i #initial condition for loop
            while partner==i: #find a new partner with a different genre
                partner = random.choice(sub_genre_sample_dict[dict_key][rh_genre]) #all other reference indices of this genre
                if partner in neg_partners: #did we put this on the right side of this left sample already?
                    partner=i #if so, keep the loop going
                elif n_test != 0:
                    if partner in test_idxs:
                        partner=i # continue the loop if this index is assigned to test data, we dont want
                               # test data indices on the left OR right side of training samples
                else:
                    neg_partners.append(partner) #otherwise put it in the list and we'll exit the loop
            for j in range(0, seq_len):
                input_neg.append(ref_samples[partner][j]) #fill partner as right hand sample
            neg_label = [0, lh_genre, rh_genre] #the labels for training, the 1 means same genre
            training_samples.append(input_neg) #add this input to the final list of training inputs
            training_labels.append(neg_label) #add corresponding negative label vector

        if(i==0 and verbose):
            print("after i==0, training_samples has "+str(len(training_samples))+" samples, which should be "+str(2*num_copies)+ " and each sample has length "+str(len(training_samples[0]))+ " which each have length" +str(len(training_samples[0][0]))+".\n")
            print("also, the label vectors are pos: "+str(pos_label)+" and neg: "+str(neg_label)+".\n\n")


    #now every element of reference_samples should have num_copies many positive and negative partners in training_samples
    #  training_labels[i] is a vector of [{0,1}, lh_genre, rh_genre] for training_sample[i]

    if(verbose):
        print("Training_samples has length "+str(len(training_samples))+", and each element has length "+str(len(training_samples[0]))+". Each of those has length "+str(len(training_samples[0][0]))+".\n\n")

        if n_test != 0:
            print("We collected " + str(
                test_count) + " leftsamples for test data. The counts in the test data for each genre are " + str(
                test_genre_count) + ".\n")
            for subj in test_data_dict.keys():
                print("Subject "+str(subj)+" has "+str(len(test_data_dict[subj]))+" many leftsamples in the test data.\n")
    #save training_samples and training_labels
    time = date.today()
    this_dir = opengenre_preproc_path+"training_data/seeded/"+str(time)+"/"
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)
    count = 0

    #set the last part of the filename by checking what already exists
    this_file = this_dir+str(hemisphere)+"_seededtrainsamplesorig"+str(count)+".p"
    while os.path.exists(this_file):
        count+=1
        this_file = this_dir + str(hemisphere) + "_seededtrainsamplesorig" + str(count) + ".p"

    #save training data and labels
    with open(this_file,"wb") as samples_fp:
        pickle.dump(training_samples,samples_fp)
    with open(this_dir+str(hemisphere)+"_seededtrainlabelsorig"+str(count)+".p","wb") as labels_fp:
        pickle.dump(training_labels,labels_fp)

    #save the test data and labels
    if n_test != 0:
        with open(this_dir+str(hemisphere)+"_testdata"+str(count)+".p","wb") as testdata_fp:
            pickle.dump(test_data_dict, testdata_fp)
        with open(this_dir+str(hemisphere)+"_testlabels"+str(count)+".p","wb") as testlabels_fp:
            pickle.dump(test_label_dict,testlabels_fp)

    #save metadata
    with open(this_dir+str(hemisphere)+"_metadata"+str(count)+".txt","w") as meta_fp:
        meta_fp.write("Creating training data with no runs held out and seeded RNG to compare to original code.\n"+
                "\nnum_samples:"+str(len(training_samples))+
                "\nallowed_genres:"+str(allowed_genres)+
                "\nthreshold:"+str(threshold)+
                "\nhemisphere:"+str(hemisphere)+
                "\nseq_len:"+str(seq_len)+
                "\nnum_copies:"+str(num_copies)+
                "\ntest_copies:"+str(test_copies)+
                "\nbinary:"+str(binary)+
                "\nmulticlass:"+str(multiclass)+
                "\ncount:"+str(count)+
                "\nvoxel_dim:"+str(voxel_dim)+
                "\n")

if __name__=="__main__":
    #seq_len should be either 5 or 10
    #hemisphere is left or right
    #threshold only 23 right now
    #num_copies is the number of positive and negative training samples to create from each left-hand sample
    #allowed_genres is a list of what it sounds like, remember range doesn't include right boundary
    make_pretraining_data("23", hemisphere="left", seq_len=5, num_copies=1, standardize=1, detrend="linear", within_subjects=1, allowed_genres=range(0,10), test_split=0)