import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import itertools
import os
import pickle
import pandas as pd
from Constants import *
import random
from random import randint
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from math import sqrt
import copy
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline

cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

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
def mask_flatten_combine_opengenre(hemisphere, threshold, set="opengenre", include_token_dims=1, verbose=1):
    # Loop through subjects
    sublist=None
    if(set=="pitchclass"):
        sublist= ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665",
               "1668", "1672", "1678", "1680"]
    elif(set=="opengenre"):
        sublist=["001", "002", "003", "004", "005"]
    for sub in sublist:
        if(set=="pitchclass"):
            sub="00"+sub
        #opengenre_preproc_path is defined in Constants.py
        #subdir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub + "/"
        subdir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid00" + sub + "/"

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
        runs_dict=runs_dicts[set]
        for task in runs_dict.keys(): #runs_dict is located in Constants.py
            max_run = runs_dict[task]

            #loop over all runs for that task
            for run in range(1, max_run+1):
                if run < 10:
                    runstr = "0"+str(run) #tsv files have left-padded run number
                else:
                    runstr = str(run)

                # Load the (65,77,65,T) preprocessed bold data
                if(set=="opengenre"):
                    filedir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub +\
                          "/dt-neuro-func-task.tag-"+task+".tag-preprocessed.run-"+runstr+".id/"
                    filepath = filedir+"bold_resampled.nii.gz" #opengenre had to be resampled down to correct space
                else:
                    filedir = pitchclass_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub +\
                          "/dt-neuro-func-task.tag-"+task+".tag-preprocessed.run-"+runstr+".id/"
                    filepath = filedir+"bold.nii" #pitchclass data is already in MNI152Lin2009, didn't need to be resampled
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



def get_accuracy(y_pred, y_true, task, log=None):
    # print("y_pred has shape "+str(y_pred.shape))
    # print("y_true has shape "+str(y_true.shape))
    cos_sim_tasks=["reconstruction"] #list of tasks where accuracy is just cosine similarity
    if task in cos_sim_tasks:
        acc=cos(y_pred,y_true)
        acc=torch.mean(acc)
        print("it's a cos_sim task, cosine sim is "+str(acc))
    else:
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
def train_val_dataset(dataset, val_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    return datasets
#applies masks AND fills ytrue_multi_batch, the ground truth list for the non-binary task. the fact that it's named "multi" distinguishes it from "binary" and is legacy naming from when it was binary class vs multi class, but now it's just a stand-in for the mask task, which isn't necessarily classification at all
def apply_masks(x, y, ref_samples, hp_dict, mask_variation, ytrue_multi_batch, sample_dists, ytrue_dist_multi1, ytrue_dist_multi2, batch_mask_indices, sample_mask_indices, mask_task, log, heldout=False):

    if (mask_variation):
        # TRAINING samples per subject
        held_idx = hp_dict["heldout_run"]
        first=True
        idx_range = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
        # retrieve smuggled information and reset to correct CLS values in those dimensions
        sub_id_int = x[0][0]
        # where is this leftsample in ref_samples
        ref_idx_left = x[0][1]
        # where is this rightsample in ref_samples
        ref_idx_right = x[0][2]
        # bounds of ref_samples for this subject
        ref_sps = len(ref_samples) // hp_dict["num_subjects"]
        ref_sub_start = ref_sps * (sub_id_int - 1)
        ref_sub_end = ref_sub_start + 1080
        # where does the heldout run start and end in this subject's segment
        if(held_idx is not None):
            ref_held_start = ref_sub_start + 120 + 80 * (held_idx)
            ref_held_end = ref_held_start + 80
        else:
            ref_held_start = None
            ref_held_end = None
        x[0][0] = 1
        x[0][1] = 0
        x[0][2] = 0

        flip = random.randint(1, 100)
        if (flip <= 50):
            k = 1
        else:
            k = 2
        idxs = random.sample(idx_range, k)
        for idx in idxs: #the things we're replacing, either one or two (k)
            sample_mask_indices.append(idx)
            if idx<6: #if it's in the left sample
                ref_idx=ref_idx_left
            else:
                ref_idx=ref_idx_right
            ref_idx=int(ref_idx)
            r_action=random.randint(1,100)
            #if r_action<=10 do nothing
            if 10<r_action<=20: #replace with another TR from the same subject, but not next to it in refsamples
                r_choice=ref_idx
                while r_choice==ref_idx:
                    if(heldout): # select a replacement from this subject's heldout run
                        r_choice=random.randint(ref_held_start, ref_held_end-1)
                    else:
                        r_choice=random.randint(ref_sub_start, ref_sub_end-1)
                        if ref_held_start<=r_choice<ref_held_end: #if we chose something in the heldout run
                            r_choice=ref_idx #repeat the loop
                    #this applies to heldout and not heldout, dont want neighbors
                    if (r_choice==ref_idx+1) or (r_choice==ref_idx-1): #if it's a neighbor in time
                        r_choice=ref_idx #repeat the loop
                #if we got this far, we got a valid replacement index
                # now just pick which element of that sequence to grab
                r_slot=random.randint(0,len(ref_samples[0])-1)
                for i in range(0, len(x[0])): #equivalently, range(0, voxel_dim)
                    # log.write("idx: "+str(idx)+"\n"+
                    #           "i: "+str(i)+"\n"+
                    #           "r_choice: "+str(r_choice)+"\n"+
                    #           "r_slot: "+str(r_slot)+"\n"+
                    #           "len(x): "+str(len(x))+"\n"+
                    #           "len(x[idx]): "+str(len(x[idx]))+"\n")
                    # log.write("len(ref_samples): "+str(len(ref_samples))+"\n")
                    # log.write("len(ref_samples[r_choice]): "+str(len(ref_samples[r_choice]))+"\n")
                    # log.write( "len(ref_samples[r_choice][r_slot]: "+str(len(ref_samples[r_choice][r_slot])))
                    x[idx][i]=ref_samples[r_choice][r_slot][i] #copy in the replacement TR
            elif r_action>20: #just make it a MSK token, which has 1 in dimension 1
                for i in range(0, len(x[0])):
                    x[idx][i]=0 #zero it out
                x[idx][1]=1
            #masking/replacing is done for this idx, now add label information to list of labels for this batch
            if(idx>6):
                ytrue_multi_idx=int(y[2])
            else:
                ytrue_multi_idx=int(y[1]) #genre labels for left/right sample

            if first:
                first=False
                #sample_dists.append(ytrue_dist_multi1.tolist())
                if(mask_task=="genre_decode"):
                    ytrue_dist_multi1[ytrue_multi_idx] = 1  # put mass on that genre label
                    ytrue_multi_batch.append(ytrue_dist_multi1.tolist())
                elif(mask_task=="reconstruction"):
                    #print("ref index is "+str(ref_idx)+" and idx and idx%6-1 are "+str(idx)+", "+str((idx%6)-1))
                    #get the voxel_dim length vector of the TR that was replaced, this is the target for this sample
                    #remember ref samples vectors only have length 5 and dont have CLS or SEP, hence mod 6 and minus 1
                    ytrue_multi_batch.append(copy.deepcopy(ref_samples[ref_idx][(idx%6)-1]))

                else:
                    log.write("illegal value for mask task in apply masks, got "+str(mask_task)+", quitting...\n")
            else:
                ytrue_dist_multi2[ytrue_multi_idx]=1 #if this is the second replacement/mask use the second distribution
                #sample_dists.append(ytrue_dist_multi2.tolist())
                if(mask_task=="genre_decode"):
                    ytrue_multi_batch.append(ytrue_dist_multi2.tolist())
                elif(mask_task=="reconstruction"):
                    #get the voxel_dim length vector of the TR that was replaced, this is the target for this sample
                    ytrue_multi_batch.append(copy.deepcopy(ref_samples[ref_idx][(idx%6)-1]))

                else:
                    log.write("illegal value for mask task in apply masks, got "+str(mask_task)+", quitting...\n")

        #end of for (idx in idxs)
        if(k==1):
            # these two lists  need to have length two even if only one replacement is done, for consistent dimensions
            sample_mask_indices.append(-1)
            #sample_dists.append([-1,0,0,0,0,0,0,0,0,0])
        #add label information for this sample to batch list
        #ytrue_multi_batch.append(sample_dists)

    else:
        mask_choice = randint(1, 10)  # pick a token to mask
        if (mask_choice >= 6):
            mask_choice += 1  # dont want to mask the SEP token at index 6, so 6-10 becomes 7-11
            # each element in the batch has 3 values, same_genre boolean, first half genre, second half genre
            # so if we're masking an element of the second half, the genre decoding label should be that half's genre
            ytrue_multi_idx = y[2]
        else:
            ytrue_multi_idx = y[1]
        ytrue_dist_multi1[ytrue_multi_idx] = 1  # set all the probability mass on the true index
        #sample_dists.append(ytrue_dist_multi1.tolist())
        #ytrue_multi_batch.append(sample_dists)
        if(mask_task=="genre_decode"):
            ytrue_multi_batch.append(ytrue_dist_multi1)
        elif(mask_task=="reconstruction"):
            #the label is just the TR image that we chose to mask
            ytrue_multi_batch.append(copy.deepcopy(x[mask_choice]))
        else:
            log.write("illegal value for mask task in apply masks, got " + str(mask_task) + ", quitting...\n")

        for i in range(0, len(x[0])):
            x[mask_choice][i] = 0  # zero it out
        x[mask_choice][1]=1
        #x[mask_choice] = torch.clone(MSK)
        sample_mask_indices.append(mask_choice)
    #add mask indices for this one sample to batch list
    batch_mask_indices.append(sample_mask_indices)



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

