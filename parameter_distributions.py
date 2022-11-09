# check the distribution of parameter values for trained and untrained models
# need to know if it's just the general range of values that induces finetuning power rather than actual learning

import torch
import torch.nn as nn
import random
from random import randint
import numpy as np
from helpers import *
from transfer_transformer import *
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from Constants import *
import sys
import os
import datetime
from torchvision import models
from torchsummary import summary


pretrain_task="both"
pretrain_idx="2"
BATCH_SIZE=1
LR_def=0.0001 #defaults, should normally be set by command line
voxel_dim=420
src_pad_sequence = [0] * voxel_dim

#currently only set up for local laptop, not discovery

pretrained_model_states = "/Volumes/External/opengenre/official/" + pretrain_task + "/states_" + pretrain_idx + ".pt"
data_path = "/Volumes/External/pitchclass/finetuning/sametimbre/datasets/2/"

# load the training and validation data
with open(data_path+"all_X.p", "rb") as samples_fp:
    train_X = pickle.load(samples_fp)
with open(data_path+"all_y.p", "rb") as labels_fp:
    train_y_detailed = pickle.load(labels_fp)
with open(data_path+"all_val_X.p", "rb") as valsamples_fp:
    val_X = pickle.load(valsamples_fp)
with open(data_path+"all_val_y.p", "rb") as vallabels_fp:
    val_y_detailed = pickle.load(vallabels_fp)

# the labels are detailed tuples, so let's just extract the True/False value into a list that we can turn into a tensor
train_y = []
val_y = []

for detailed_label in train_y_detailed:
    tf = detailed_label[0] # True/False is the first thing in the tuple
    if tf==True:
        train_y.append([1])
    elif tf==False:
        train_y.append([0])
    else:
        print("label value neither True nor False in detailed label "+str(detailed_label)+", quitting...")
        exit(0)

for detailed_val_label in val_y_detailed:
    tf = detailed_val_label[0] # True/False is the first thing in the tuple
    if tf==True:
        val_y.append([1])
    elif tf==False:
        val_y.append([0])
    else:
        print("label value neither True nor False in detailed val label "+str(detailed_val_label)+", quitting...")
        exit(0)

# convert to tensors and create objects pytorch can use
train_X = np.array(train_X)
train_y = np.array(train_y)
val_X = np.array(val_X)
val_y = np.array(val_y)

train_X = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_y)
val_X = torch.from_numpy(val_X)
val_y = torch.from_numpy(val_y)

finetune_data = TrainData(train_X, train_y)
val_data = TrainData(val_X, val_y)

# BATCH_SIZE is defined at the top of this file
train_loader = DataLoader(dataset=finetune_data, batch_size=BATCH_SIZE, shuffle=True)  # make the DataLoader object
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

# create the model
model = Transformer(next_sequence_labels=2, num_genres=10, src_pad_sequence=src_pad_sequence, max_length=12, voxel_dim=voxel_dim, ref_samples=None, mask_task="reconstruction", print_flag=0)
model.load_state_dict(torch.load(pretrained_model_states), strict=False)
model = model.float()
total=0
count=0


print(model)
for layer in model.children():
    print(layer._get_name())
    weights=list(layer.parameters())
    print(weights)
    # for weight in weights:
    #     #print(weight)
    #     temp = torch.mean(weight)
    #     total+=temp
    #     count+=1
    #     print(weight.)
    #     print(temp)

    break

# avg = total/count
# print("average is "+str(avg))
