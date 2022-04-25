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
from datetime import datetime
from torch.utils.data import Dataset

# make the pretraining datasets according to several parameters
#  the first is the probability threshold for inclusion in STG as a string, e.g "23", the second is "left" or "right"
#   the third is the length of each half of the sample, the fourht is the number of repetitions we want from the -test- runs, from 1 to 4. Recall that each test run is the same 10
#    clips repeated four times. Default is 1, i.e each set of 10 only once/only the first ten from each test run.
#     The fifth and sixth determine the binary and multiclass classification tasks we want to train on.
def make_pretraining_data(threshold, hemisphere, seq_len=5, test_copies=1, binary="same_genre", multiclass="genre_decode", verbose=1):
    #runs_dict is defined in Constants.py
    test_runs = runs_dict["Test"]
    training_runs = runs_dict["Training"]

    #size of left or right STG voxel space, with 3 token dimensions already added in
    voxel_dim = COOL_DIVIDEND+3 #defined in Constants.py
    CLS = [1] + ([0] * (voxel_dim - 1))  # first dimension is reserved for cls_token flag
    MSK = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
    SEP = [0, 0, 1] + ([0] * (voxel_dim - 3))  # third dimension is reserved for sep_token flag

    #spr stands for samples per run, the right hand side is defined in Constants.py
    spr = opengenre_samples_per_run

    #each element of this list should be (12,real_voxel_dim+3), i.e CLS+5TRs+SEP+5TRs
    #   where 3 extra dimensions have been added to the front of the voxel space for the tokens
    training_samples = []

    #each element of this list should be [m,n] where m is in {0,1} and n is in {1,2,...,10}
    training_labels = []

    #keep count of how many sample/label pairs we've created
    count = 0

    #a list of indices for each genre, where to find each genre in the aggregate training data list
    genre_sample_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}

    #loop through subjects
    for sub in ["001", "002", "003", "004", "005"]:
        iter=0 #iterations of the next loop, resets per subject
        #opengenre_preproc_path is defined in Constants.py
        subdir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub + "/"
        #load voxel data and labels
        with open(subdir+"STG_allruns"+hemisphere+"_t"+threshold+".p", "rb") as data_fp:
            voxel_data = pickle.load(data_fp)
        with open(subdir+"labelindices_allruns.p","rb") as label_fp:
            labels = pickle.load(label_fp)

        timesteps = len(voxel_data)
        #we're going to obtain timesteps//seq_len many samples from each subject
        for t in range(0, timesteps//seq_len):
            start = t*seq_len
            this_sample = [CLS]
            # add the seq_len many next images
            for s in range(0, seq_len):
                this_sample.append(voxel_data[start+s])
            #add separator token
            this_sample.append(SEP)
            #magic numbers at the moment, each label corresponds to 10 TRs and we have a seq_len of 5
            #  so cut the count in half to get the right label
            this_genre = labels[iter//2]

            #add count as an index corresponding to this genre
            genre_sample_dict[this_genre].append(count)
            same_genre_choice = randint(0,1)
            #the labels for training, same genre boolean and genre index
            this_label = [same_genre_choice, this_genre]
            #append to aggregate lists
            training_samples.append(this_sample)
            training_labels.append(this_label)
            #increase count
            count = count+1
            iter = iter+1
    # looping over subjects is done
    # the aggregate lists are complete, let's get the second half of each sample
    for i in range(0, len(training_samples)):
        same_genre = training_labels[i][0]
        this_genre = training_labels[i][1]

        if(same_genre):
            potentials = genre_sample_dict[this_genre] #a list of indices in training_samples with this genre
        else:
            choice = this_genre
            while (choice==this_genre): #get a different genre index
                choice=randint(0,9)
            potentials = genre_sample_dict[choice] #a list of indices in training_samples with a different genre

        partner = i
        while(partner==i): #get a partner different from self
            partner_choice=randint(0,len(potentials)-1) #pick something in the list of potentials
            partner=potentials[partner_choice] #get the index in training_samples corresponding to that choice

        for j in range(0,seq_len):
            #training_samples[i] is the current sample
            #training_samples[partner] is the randomly chosen sample with same/different genre
            training_samples[i].append(training_samples[partner][j])
        training_labels[i].append(training_labels[partner][1]) #get the genre index of the partner

    #now every elemet of training_samples should have its partner, and their corresponding entry in
    #  training_labels includes the genre of the partner

    if(verbose):
        print("Training_samples has length "+str(len(training_samples))+", and each element has length "+str(len(training_samples[0]))+". Each of those has length "+str(len(training_samples[0][0]))+".\n\n")

    #save training_samples and training_labels
    time = datetime.today()
    this_dir = opengenre_preproc_path+"training_data/"+str(time)+"/"
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)
    count = 0

    #set the last part of the filename by checking what already exists
    this_file = this_dir+str(hemisphere)+"_samples"+str(count)+".p"
    while os.path.exists(this_file):
        count+=1
        this_file = this_dir + str(hemisphere) + "_samples" + str(count) + ".p"

    with open(this_file,"wb") as samples_fp:
        pickle.dump(training_samples,samples_fp)
    with open(this_dir+str(hemisphere)+"_labels"+str(count)+".p","wb") as labels_fp:
        pickle.dump(training_labels,labels_fp)
    with open(this_dir+"metadata"+str(count)+".txt","w") as meta_fp:
        meta_fp.write("threshold:"+str(threshold)+
                      "\nhemisphere:"+str(hemisphere)+
                      "\nseq_len:"+str(seq_len)+
                      "\ntest_copies:"+str(test_copies)+
                      "\nbinary:"+str(binary)+
                      "\nmulticlass:"+str(multiclass)+
                      "\ncount:"+str(count)+
                      "\nvoxel_dim:"+str(voxel_dim)+
                      "\n")

if __name__=="__main__":
    make_pretraining_data(23, )