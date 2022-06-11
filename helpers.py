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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from math import sqrt
import copy
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline

class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

# create pickled lists of genre labels as strings and indices based on the events.tsv files from opengenre dataset
def make_opengenre_labels():
    # Loop through subjects
    # The order of clips is the same for each subject(?) so we don't really need to read each subject's tsv files separately
    # But we do need to pickle the list for each subject separately so whatever.
    for sub in ["001", "002", "003", "004", "005"]:
        filedir = opengenre_events_path+"sub-"+sub+"/func/"

        #loop over both tasks
        for task in runs_dict.keys(): #runs_dict is located in Constants.py
            max_run = runs_dict[task]

            #loop over all runs for that task
            for run in range(1, max_run+1):
                #create lists to be pickled
                string_l = []
                idx_l = []
                if run < 10:
                    runstr = "0"+str(run) #tsv files have left-padded run number
                else:
                    runstr = str(run)
                filepath = filedir + "sub-"+sub+"_task-"+task+"_run-"+runstr+"_events.tsv"

                # get tsv file
                events_df = pd.read_csv(filepath, sep='\t')
                genres = events_df["genre"] #it works like a dictionary
                for clip in genres.keys(): #index 0 is dummy listening/padding, not a real sample and should be excluded
                    if (clip !=0):
                        this_genre = genres[clip].strip("\'") #genres have single quotes in the dictionary, get rid of them
                        string_l.append(this_genre)
                        idx_l.append(genre_dict[this_genre]) #genre_dict defined in Constants.py

                        # Let's save these lists to the same place as our preprocessed bold data
                        temp_path = opengenre_preproc_path+"sub-sid"+sub+"/sub-sid"+sub+"/"+\
                                    "dt-neuro-func-task.tag-"+task.lower()+".tag-preprocessed.run-"+runstr+".id/"
                        # Save the list of strings
                        fp = open(temp_path+"genrestring_list.p","wb")
                        pickle.dump(string_l, fp)
                        fp.close()
                        # Save the list of indices
                        fp = open(temp_path+"genreindex_list.p","wb")
                        pickle.dump(idx_l, fp)
                        fp.close()

# the above function makes separate lists for each run, this function stacks them for each subject
def stack_opengenre_labels(verbose=1):
    #loop through subjects
    for sub in ["001", "002", "003", "004", "005"]:
        #opengenre_preproc_path is defined in Constants.py
        subdir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub + "/"

        #list for combinining all runs
        stacked_strings = []
        stacked_indices = []

        #loop over both tasks
        for task in runs_dict.keys(): #runs_dict is located in Constants.py
            max_run = runs_dict[task]

            #loop over all runs for that task
            for run in range(1, max_run+1):
                if run < 10:
                    runstr = "0"+str(run) #tsv files have left-padded run number
                else:
                    runstr = str(run)

                # Load the labels for this subject/task/run
                temp_path = opengenre_preproc_path + "sub-sid" + sub + "/sub-sid" + sub + "/" + \
                            "dt-neuro-func-task.tag-" + task.lower() + ".tag-preprocessed.run-" + runstr + ".id/"

                string_fp = open(temp_path + "genrestring_list.p", "rb")
                idx_fp = open(temp_path+"genreindex_list.p","rb")
                stringlist = pickle.load(string_fp)
                idxlist = pickle.load(idx_fp)

                #add this run's labels to the aggregate list
                for lbl in stringlist:
                    stacked_strings.append(lbl)
                for lbl in idxlist:
                    stacked_indices.append(lbl)

                #close file pointers
                string_fp.close()
                idx_fp.close()

        allstrings_fp = open(subdir+"labelstrings_allruns.p","wb")
        allindices_fp = open(subdir+"labelindices_allruns.p","wb")

        if(verbose):
            print("Collected "+str(len(stacked_strings))+" string labels and "+str(len(stacked_indices))+" index labels for sub-sid"+sub+". Saving...\n")

        pickle.dump(stacked_strings, allstrings_fp)
        pickle.dump(stacked_indices, allindices_fp)

        if(verbose):
            print("Completed sub-sid"+sub+".\n\n")


#apply binary masks that are already on disk at opengenre_preproc-path/sub/sub/STG_masks
#flatten to 1 dimension from 3D MNI space
#combine all 8 runs
#hemisphere is either "left" or "right"
#include token dims tells us whether to insert 3 extra dimensions at the front of the lists to be used by pretraining tokens
#  i.e if you are going to be pretraining with fmribert, this should be set to True.
def mask_flatten_combine_opengenre(hemisphere, threshold, include_token_dims=1, verbose=1):
    # Loop through subjects
    for sub in ["001", "002", "003", "004", "005"]:
        #opengenre_preproc_path is defined in Constants.py
        subdir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub + "/"

        #load binary mask
        mask_fp = open(subdir + "STG_masks/STGbinary_"+hemisphere+"_t"+threshold+".p","rb")
        mask = pickle.load(mask_fp)

        #the mask is (65,77,65) so rather than keeping that all in memory, let's get a list of the coordinates we want to extract
        voxel_list = []
        count = 0 #variable to count the included voxels
        for x in range(0,65):
            for y in range(0,77):
                for z in range(0,65):
                    if (mask[x][y][z]==1):
                        voxel_list.append((x,y,z))
                        count+=1
        if(verbose):
            print("For sub-sid"+str(sub)+", included count was "+str(count)+".\n")
        mask_fp.close()

        #save voxel list just in case
        with open(subdir + "STG_masks/voxellist_"+hemisphere+"_t"+threshold+".p","wb") as voxel_fp:
            pickle.dump(voxel_list,voxel_fp)

        #optional print messages, default is True
        if(verbose):
            print("For sub-sid" + str(sub) + ", we are extracting " + str(len(voxel_list)) + " voxels.\n")
            print("Saved voxellist for sub-sid"+str(sub)+".\n")

        #list for combinining all runs
        masked_data = []
        #loop over both tasks
        for task in runs_dict.keys(): #runs_dict is located in Constants.py
            max_run = runs_dict[task]

            #loop over all runs for that task
            for run in range(1, max_run+1):
                if run < 10:
                    runstr = "0"+str(run) #tsv files have left-padded run number
                else:
                    runstr = str(run)

                # Load the (65,77,65,T) preprocessed bold data
                filedir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub +\
                          "/dt-neuro-func-task.tag-"+task+".tag-preprocessed.run-"+runstr+".id/"
                filepath = filedir+"bold_resampled.nii.gz"
                bold_img = nib.load(filepath)
                bold_data = bold_img.get_fdata()

                # steps 0 through 9 inclusive are dummy data
                for t in range(10,410):
                    #array to hold flattened ROI data at timestep t
                    # add three extra dimensions for pretraining tokens if flag is set
                    if(include_token_dims):
                        flattened_t = [0,0,0]
                    else:
                        flattened_t = []
                    #each element of voxel_list is a tuple of coordinates in 3d space
                    for voxel in voxel_list:
                        vx, vy, vz = voxel[0],voxel[1],voxel[2]

                        flattened_t.append(bold_data[vx][vy][vz][t])
                    masked_data.append(flattened_t)
        # end of for task in task_dict
        # all 18 runs should be in masked_data now, len should be 18*400
        if(verbose):
            print("Length of masked_data for sub-sid"+sub+" is "+str(len(masked_data))+".\n")
            print("Each element of masked_data has length "+str(len(masked_data[0]))+".\n")

        # save masked_data
        with open(subdir+"STG_allruns"+hemisphere+"_t"+threshold+".p","wb") as stacked_fp:
            pickle.dump(masked_data, stacked_fp)

        if(verbose):
            print("sub-sid"+sub+" complete.\n\n")



def get_accuracy(y_pred, y_true, log=None):
    # print("y_pred has shape "+str(y_pred.shape))
    # print("y_true has shape "+str(y_true.shape))
    prediction_idxs = torch.argmax(y_pred,dim=1)
    true_idxs = torch.argmax(y_true, dim=1)

    correct_sum = (prediction_idxs == true_idxs).sum().float()
    # if(log is  not None):
    #     log.write("prediction idx are " + str(prediction_idxs))
    #     log.write("true idxs are " + str(true_idxs))
    #     log.write("correct sum is "+str(correct_sum))
    acc = correct_sum/y_true.shape[0]
    acc = torch.round(acc*100)
    # if(log is not None):
    #     log.write("accuracy for this batch was "+str(acc))
    return acc

#treat each voxel as a channel
#calculate mean of each channel across timesteps and subtract it from each timestep
#calculate new standard deviation of each channel and divide each timestep by that amount
#voxel data should already be TRs by COOL_DIVIDEND after accounting for test_copies
def standardize_flattened(voxel_data):
    stand = copy.deepcopy(voxel_data)

    TRs = len(voxel_data)
    channels = len(voxel_data[0])
    for channel in range(3, channels): #don't do this for first three channels, they're just token flag dimensions

        mean = 0
        for TR in range(0, TRs):
            mean+=voxel_data[TR][channel]
        mean=mean/TRs #total sum of channel divided by number of timesteps

        new_mean = 0
        for TR in range(0, TRs):
            temp = stand[TR][channel] - mean #subtract out the mean of this channel
            stand[TR][channel] = temp
            new_mean += temp


        new_mean=new_mean/TRs #mean of shifted data
        new_dev = 0
        for TR in range(0, TRs):
            new_dev+=(stand[TR][channel]-new_mean)**2
        new_dev = sqrt(new_dev/TRs)

        #unit variance
        for TR in range(0, TRs):
            stand[TR][channel]= (stand[TR][channel]/new_dev)

    return stand

def detrend_flattened(voxel_data, detrend="linear"):
    # each voxel is detrended independently so fix the voxel i.e the column and loop over the timesteps
    n_columns = len(voxel_data[0])
    n_rows = len(voxel_data)
    x_train = range(0, n_rows)
    x_train=np.array(x_train)
    voxel_data = np.array(voxel_data)
    detrend_data=np.zeros(voxel_data.shape)
    n_plotpoints = 7200
    y_plots=[]
    model=None
    # dimensions 0, 1, and 2 in voxel space are just token dimensions, don't detrend those
    for voxel in range(3, n_columns):
        y_train = voxel_data[:,voxel]
        maxv = max(x_train)
        minv = min(x_train)

        X_train = x_train[:, np.newaxis]

        # either train a linear regressor as michael recommends
        #  or train a cubic spline as concluded by Tanabe et al. (2002)
        #   the detrend variable is given as a parameter to the parent function, make_pretraining_data
        if(detrend=="linear"):
            model = make_pipeline(PolynomialFeatures(1), Ridge(alpha=1e-3))
            model.fit(X_train, y_train)

        # in this case, the variable looks like "splineXY" with X the knots and Y the degree
        #  generally Y should only be 3, i.e cubic spline
        elif(detrend[:6]=="spline"):

            knots=int(detrend[6]) #number of points to get smooth degree derivatives around
            thedegree=int(detrend[7])
            model = make_pipeline(SplineTransformer(n_knots=knots, degree=thedegree), Ridge(alpha=1e-3))
            model.fit(X_train, y_train)

        y_pred = model.predict(X_train) #get the values of the regression line

        y_resid = y_train - y_pred #get the residuals by subtracting off the regression line
        for j in range(0, n_rows):
            detrend_data[j][voxel]=y_resid[j]

    # detrending should be done
    return detrend_data.tolist()




if __name__=="__main__":
    # create pickled lists of genre labels as strings and indices based on the events.tsv files from opengenre dataset
    #make_opengenre_labels()
    # stack labels for all runs
    #stack_opengenre_labels()

    #create pickled lists of pre-training data from opengenre datasets
    # uses the binary masks to grab data, fixes x first then y then z, so flattens in the reverse order
    #  combines all 8 runs for each subject, note first 10 of 410 TRs in each run are dummy data and not used
    #   pass in the hemisphere and the probability inclusion threshold as a string, and optional verbose parameter, default is True
    #mask_flatten_combine_opengenre("left", "23")
    #mask_flatten_combine_opengenre("right", "23")
    #make_pretraining_data("23", "left")
    print("none")

def train_val_dataset(dataset, val_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    return datasets