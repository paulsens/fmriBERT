import torch
import torch.nn as nn
import random
from random import randint
import numpy as np
from helpers import *
from voxel_transformer import *
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from Constants import *
import sys
import os
import datetime

pretrain_idx = 0
pretrain_task = "both"
voxel_dim=420
src_pad_sequence = [0] * voxel_dim
BATCH_SIZE = 1

if __name__=="__main__":

    hp_dict = {
        "hemisphere": "left",
        "data_dir": "2022-06-12",
        "BATCH_SIZE":1,
        "count":str(pretrain_idx),
        "mask_task":"reconstruction",
    }
    hp_dict["data_path"] = opengenre_preproc_path + "training_data/" + hp_dict["data_dir"] + "/"

    # env is defined in Constants.py
    if env=="local":
        pretrained_model_states = "/Volumes/External/opengenre/final/"+pretrain_task+"/states_"+str(pretrain_idx)+".pt"
        data_path = "/Volumes/External/pitchclass/finetuning/sametimbre/datasets/2/"
    if env=="discovery":
        pretrained_model_states = "/isi/music/auditoryimagery2/seanthesis/opengenre/official/"+pretrain_task+"/states_"+pretrain_idx+".pt"
        data_path = "/isi/music/auditoryimagery2/seanthesis/pitchclass/finetuning/sametimbre/datasets/5/"

    with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_valsamples" + hp_dict["count"] + ".p",
              "rb") as samples_fp:
        val_X = pickle.load(samples_fp)
    with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_vallabels" + hp_dict["count"] + ".p", "rb") as labels_fp:
        val_Y = pickle.load(labels_fp)

    val_X = np.array(val_X)

    val_Y = np.array(val_Y)
    val_X = torch.from_numpy(val_X)
    val_Y = torch.from_numpy(val_Y)
    val_data = TrainData(val_X, val_Y)  # make TrainData object for validation data
    MSK_token = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
    MSK_token = np.array(MSK_token)
    MSK_token = torch.from_numpy(MSK_token)


    # create the model
    model = Transformer(next_sequence_labels=2, num_genres=10, src_pad_sequence=src_pad_sequence, max_length=12, voxel_dim=voxel_dim, ref_samples=None, mask_task=hp_dict["mask_task"], print_flag=0)

    val_loader = DataLoader(dataset=val_data, batch_size=hp_dict["BATCH_SIZE"], shuffle=False)
    #print(model.encoder.layers[0].attention.values_heads[0].weight)

    # if we want to load pretrained weights:
    model.load_state_dict(torch.load(pretrained_model_states), strict=False)
    #print(model.encoder.layers[0].attention.values_heads[0].weight)

    # the first few mask assignments for the val split
    #[[11, -1]], [[11, -1]], [[7, -1]], [[7, -1]], [[11, 2]], [[11, 2]], [[5, 10]], [[5, 10]]

    model.eval()
    count = 0
    with torch.no_grad():

        for X_batch_val, y_batch_val in val_loader:
            X_batch_val = X_batch_val.float()
            y_batch_val = y_batch_val.float()
            if count == 0:
                batch_mask_indices_val = [[11, -1]]
                X_batch_val[0][11] = MSK_token

            elif count ==1:
                batch_mask_indices_val = [[7, -1]]
                X_batch_val[0][7] = MSK_token


            X = X_batch_val[0]
            X_batch_val[0][0][0] = 1
            X_batch_val[0][0][1] = 0
            X_batch_val[0][0][2] = 0
            y_true = y_batch_val[0]

            print("Xbatchval is "+str(X_batch_val))
            ypred_bin_batch_val, ypred_multi_batch_val = model(X_batch_val, batch_mask_indices_val)

            print(ypred_bin_batch_val)
            print(ypred_multi_batch_val)
            count+=1
            if count == 2:
                quit(0)