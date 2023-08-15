from voxel_transformer import *
from pitchclass_data import *
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import pickle
import numpy as np
from random import randint

seq_len = 6
TIMESTEPS = 10000
# ds1 = torch.normal(0, 1, size=(10000,101))
# ds2 = torch.normal(10, 1, size=(10000,101))
# CLS_token = [1] + ([0] * (voxel_dim - 1))  # first dimension is reserved for cls_token flag
# MSK_token = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
# SEP_token = [0, 0, 1] + ([0] * (voxel_dim - 3))  # third dimension is reserved for sep_token flag
true_count = 0
false_count = 0
voxel_dim = 501
mydataset = []
mylabels = []
for i in range(0,10000):
    #start a new list
    temp = np.zeros((14, 504))
    #add CLS token
    temp[0][0]=1
    #add SEP token
    temp[7][2]=1
    #generate six 101-dimensional vectors with mean 0 and sd 1
    sample = torch.normal(0, 1, size=(6, 501))
    #print("sample is "+str(sample))
    for j in range(1,7):
        for k in range(3,504):
            temp[j][k] = sample[j-1][k-3]
    #flip a coin
    flip = randint(0,1)
    #if it's a true sample
    if flip == 1:
        #print("flip was "+str(flip))

        #draw second sample from same distribution as sample 1
        sample2 = torch.normal(0, 1, size=(6,501))
        true_count += 1
        #add the true label to labels array
        mylabels.append([0,1])
    else:
        #print("flip was "+str(flip))
        #otherwise draw second sample from statistically distinct distribution
        sample2 = torch.normal(8, 1, size=(6,501))
        false_count += 1
        #add the false label to labels array
        mylabels.append([1,0])
    #either way, fill in the second sample in this sample
    #print("sample 2 is " + str(sample2))

    for j in range(8, 14):
        for k in range(3, 504):
            temp[j][k] = sample2[j-8][k-3]
    #print("temp is "+str(temp))
    mydataset.append(temp.tolist())

#print("mydataset is "+str(mydataset))
print("final true count is "+str(true_count))
print("final false count is "+str(false_count))
samples_fp = open("/Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT/pitchclass/samples/gaussian_samples.p", "wb")
labels_fp = open("/Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT/pitchclass/samples/gaussian_labels.p", "wb")
pickle.dump(mydataset, samples_fp)
pickle.dump(mylabels, labels_fp)
samples_fp.close()
labels_fp.close()

