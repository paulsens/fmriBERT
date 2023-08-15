import torch
import torch.nn as nn
import random
from random import randint
import numpy as np
from helpers import *
from talking_transformer import *
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from Constants import *
import sys
import os
import datetime

voxel_dim = 420
src_pad_sequence = [0] * voxel_dim
LR = 0.00001
MASK_INDICES = "finetune" #legacy crap that needs to be replaced, should be finetune or sametimbre

best_listFalsePosNeg = [(0,10), (1,6), (2,10), (3,10), (4,10), (5,10), (6,5), (7,9), (8,10), (9,10)] # (index, epoch) to load weights
best_listTruePosNeg = [(0,9), (1,10), (2,9), (3,10), (4,8), (5,7), (6,10), (7,9), (8,10), (9,10)]
best_listLeftHeldoutSubs = [(0, 5), (1, 7), (2, 4), (3, 6), (4, 9), (5, 4), (6, 9), (7, 8)] # epochs are 0-indexed
best_listRightHeldoutSubs = [(0, 6), (1, 6), (2, 9), (3, 4), (4, 8), (5, 8), (6, 7), (7, 9)]

# bests for music genre pretraining
bests_both = [(0, 9), (1, 9), (2, 8), (3, 7), (4, 9), (5, 9), (6, 9), (7, 9), (8, 8), (9, 4), (10, 9), (11, 9)]
bests_CLS_only = [(0, 8), (1, 6), (2, 9), (3, 9), (4, 8), (5, 9), (6, 8), (7, 8), (8, 8), (9, 8), (10, 9), (11, 9)]

######## genre pretraining
# num_subjects = 17
# datasets_path = "/Volumes/External/opengenre/preproc/training_data/"
# model_path = "/Volumes/External/opengenre/preproc/trained_models/"
# fold_n = 0 # 0 to 11 inclusive
# epoch_n = 9 # 0 to 9 inclusive

## CLS_only
# dataset = datasets_path + "2022-06-12/"
# pretrained_model_states = model_path + "both/2023-03-21/states_"+str(fold_n)+str(epoch_n)+".pt
# train_X = dataset + "left_samples"+str(fold_n)+".p"
# train_y = dataset + "left_labels"+str(fold_n)+".p"
# val_X = dataset + "left_valsamples"+str(fold_n)+".p"
# val_y = dataset + "left_vallabels"+str(fold_n)+".p"

## both
# dataset = datasets_path + "2022-06-12/"
# pretrained_model_states = model_path + "binaronly/2023-03-21/states_"+str(fold_n)+str(epoch_n)+".pt
# train_X = dataset + "left_samples"+str(fold_n)+".p"
# train_y = dataset + "left_labels"+str(fold_n)+".p"
# val_X = dataset + "left_valsamples"+str(fold_n)+".p"
# val_y = dataset + "left_vallabels"+str(fold_n)+".p"

######## samegenre
# num_subjects = 17
# model_path = "/Volumes/External/opengenre/preproc/trained_models/finetuned/
# datasets_path = "/Volumes/External/opengenre/preproc/training_data/
# fold_n = 0 # 0 to 11 inclusive
# epoch_n = 9 # 0 to 9 inclusive

## sg on CLS only
# dataset = datasets_path + "2022-11-14/"
# pretrained_model_states = model_path + "sg_on_CLS_only/states_"+str(fold_n)+str(epoch_n)+".pt"
# train_X = dataset + "left_samples"+str(fold_n)+".p"
# train_y = dataset + "left_labels"+str(fold_n)+".p"
# val_X = dataset + "left_valsamples"+str(fold_n)+".p"
# val_y = dataset + "left_vallabels"+str(fold_n)+".p"

## sg on both
# dataset = datasets_path + "2022-11-14/"
# pretrained_model_states = model_path + "sg_on_both/states_"+str(fold_n)+str(epoch_n)+".pt"
# train_X = dataset + "left_samples"+str(fold_n)+".p"
# train_y = dataset + "left_labels"+str(fold_n)+".p"
# val_X = dataset + "left_valsamples"+str(fold_n)+".p"
# val_y = dataset + "left_vallabels"+str(fold_n)+".p"

## sg on fresh
# dataset = datasets_path + "2022-11-14/"
# pretrained_model_states = model_path + "fresh/states_"+str(fold_n)+str(epoch_n)+".pt"
# train_X = dataset + "left_samples"+str(fold_n)+".p"
# train_y = dataset + "left_labels"+str(fold_n)+".p"
# val_X = dataset + "left_valsamples"+str(fold_n)+".p"
# val_y = dataset + "left_vallabels"+str(fold_n)+".p"

###### audimgNTP
# num_subjects = 5
# model_path = "/Volumes/External/pitchclass/trained_models/NTP/"
# datasets_path = "/Volumes/External/pitchclass/pretraining/"
# fold_n = 4 # 0 to 7 inclusive
# epoch_n = 9
#
# # left STG
# epoch_offset = 0 # 0 is left, 10 is right
# dataset = datasets_path + "2023-08-03-5TR_2stride_Trueposneg_2heldout_leftSTG/"
# pretrained_model_states = model_path + "states_"+str(fold_n)+str(epoch_n+epoch_offset)+".pt"
# train_X = dataset + "all_X_fold"+str(fold_n)+".p"
# train_y = dataset + "all_y_fold"+str(fold_n)+".p"
# val_X = dataset + "all_val_X_fold"+str(fold_n)+".p"
# val_y = dataset + "all_val_y_fold"+str(fold_n)+".p"
# print("Inference on left STG NTP:\n")

## right STG
# epoch_offset = 10 # 0 is left, 10 is right
# dataset = datasets_path + "2023-08-03-5TR_2stride_Trueposneg_2heldout_rightSTG/"
# pretrained_model_states = model_path + "states_"+str(fold_n)+str(epoch_n+epoch_offset)+".pt"
# train_X = dataset + "all_X_fold"+str(fold_n)+".p"
# train_y = dataset + "all_y_fold"+str(fold_n)+".p"
# val_X = dataset + "all_val_X_fold"+str(fold_n)+".p"
# val_y = dataset + "all_val_y_fold"+str(fold_n)+".p"

####### sametimbre
num_subjects = 5
model_path = "/Volumes/External/pitchclass/trained_models/sametimbre/"
datasets_path = "/Volumes/External/pitchclass/finetuning/sametimbre/datasets/"
fold_n = 4 # 0 to 7 inclusive

# # right STG finetune
# dataset = datasets_path + "2023-08-03-5TR_audimg_HOruns_hasimagined_1-1-pairs_repetition1_rightSTG_2heldout/"
# pretrained_model_states = model_path + "rightHeldoutSubs/states_"+str(fold_n)+".pt"
# train_X = dataset + "all_X_fold"+str(fold_n)+".p"
# train_y = dataset + "all_y_fold"+str(fold_n)+".p"
# val_X = dataset + "all_val_X_fold"+str(fold_n)+".p"
# val_y = dataset + "all_val_y_fold"+str(fold_n)+".p"

## left STG finetune
# dataset = datasets_path + "2023-08-03-5TR_audimg_HOruns_hasimagined_1-1-pairs_repetition1_leftSTG_2heldout/"
# pretrained_model_states = model_path + "leftHeldoutSubs/states_"+str(fold_n)+".pt"
# train_X = dataset + "all_X_fold"+str(fold_n)+".p"
# train_y = dataset + "all_y_fold"+str(fold_n)+".p"
# val_X = dataset + "all_val_X_fold"+str(fold_n)+".p"
# val_y = dataset + "all_val_y_fold"+str(fold_n)+".p"

## right STG RI
# dataset = datasets_path + "2023-08-03-5TR_audimg_HOruns_hasimagined_1-1-pairs_repetition1_rightSTG_2heldout/"
# pretrained_model_states = model_path + "sametimbre/rightHeldoutSubsFresh/states_"+str(fold_n)+".pt"
# train_X = dataset + "all_X_fold"+str(fold_n)+".p"
# train_y = dataset + "all_y_fold"+str(fold_n)+".p"
# val_X = dataset + "all_val_X_fold"+str(fold_n)+".p"
# val_y = dataset + "all_val_y_fold"+str(fold_n)+".p"

# left STG RI
dataset = datasets_path + "2023-08-03-5TR_audimg_HOruns_hasimagined_1-1-pairs_repetition1_leftSTG_2heldout/"
pretrained_model_states = model_path + "leftHeldoutSubsFresh/states_"+str(fold_n)+".pt"
train_X = dataset + "all_X_fold"+str(fold_n)+".p"
train_y = dataset + "all_y_fold"+str(fold_n)+".p"
val_X = dataset + "all_val_X_fold"+str(fold_n)+".p"
val_y = dataset + "all_val_y_fold"+str(fold_n)+".p"

######## NAcc NTP
#num_subjects = 5


######## Same-Session
#num_subjects = 5



############
with open(train_X, "rb") as samples_fp:
    train_X_loaded = pickle.load(samples_fp)
with open(train_y, "rb") as labels_fp:
    train_y_loaded = pickle.load(labels_fp)
with open(val_X, "rb") as valsamples_fp:
    val_X_loaded = pickle.load(valsamples_fp)
with open(val_y, "rb") as vallabels_fp:
    val_y_loaded = pickle.load(vallabels_fp)


# need the  below if the dataset uses detailed labels
train_y_detailed = train_y_loaded
train_y_loaded = []
val_y_detailed = val_y_loaded
val_y_loaded = []
for detailed_label in train_y_detailed:
    tf = detailed_label[0]  # True/False is the first thing in the tuple
    if tf == 1:
        train_y_loaded.append([1])
    elif tf == 0:
        train_y_loaded.append([0])

for detailed_val_label in val_y_detailed:
    tf = detailed_val_label[0]  # True/False is the first thing in the tuple
    if tf == 1:
        val_y_loaded.append([1])
    elif tf == 0:
        val_y_loaded.append([0])

train_X_loaded = np.array(train_X_loaded)
train_y_loaded = np.array(train_y_loaded)
val_X_loaded = np.array(val_X_loaded)
val_y_loaded = np.array(val_y_loaded)

train_X_loaded = torch.from_numpy(train_X_loaded)
train_y_loaded = torch.from_numpy(train_y_loaded)
val_X_loaded = torch.from_numpy(val_X_loaded)
val_y_loaded = torch.from_numpy(val_y_loaded)

finetune_data = TrainData(train_X_loaded, train_y_loaded)
val_data = TrainData(val_X_loaded, val_y_loaded)

BATCH_SIZE = 1
# BATCH_SIZE is defined at the top of this file
train_loader = DataLoader(dataset=finetune_data, batch_size=BATCH_SIZE, shuffle=True)  # make the DataLoader object
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

# create the model
model = Transformer(num_CLS_labels=2, num_genres=10, src_pad_sequence=src_pad_sequence,
                    max_length=12, voxel_dim=voxel_dim, ref_samples=None, mask_task="reconstruction",
                    print_flag=0,
                    heads=2, num_layers=3,
                    forward_expansion=4)

training_tensors = model.parameters()

#################################################
model.load_state_dict(torch.load(pretrained_model_states), strict=False)
#################################################

model = model.float()
criterion_bin = nn.CrossEntropyLoss()
optimizer = optim.Adam(training_tensors, lr=LR, betas=(0.9, 0.999), weight_decay=0.0001)

# train
best_avg_val_acc = 0
best_val_epoch = -1
seed = 1
with torch.no_grad():
    for e in range(1, 2):
        # 0'th index is the number of times the model was correct when the ground truth was 0, and when ground truth was 1
        bin_correct_train = [0, 0]
        bin_correct_val = [0, 0]
        bin_neg_count_train = 0  # count the number of training samples where 0 was the correct answer
        bin_neg_count_val = 0  # count the number of validation samples where 0 was the correct answer

        random.seed(seed + e)
        torch.manual_seed(seed + e)
        np.random.seed(seed + e)
        model.train()  # sets model status, doesn't actually start training
        # need the above every epoch because the validation part below sets model.eval()
        epoch_loss = 0
        epoch_acc = 0

        batch_count = 0
        # for X_batch, y_batch in train_loader:
        #
        #     X_batch = X_batch.float()
        #     y_batch = y_batch.float()
        #
        #     # X_batch, y_batch = X_batch.to(hp_dict["device"]), y_batch.to(hp_dict["device"])
        #     ytrue_bin_batch = []  # list of batch targets for binary classification task
        #
        #     optimizer.zero_grad()  # reset gradient to zero before each mini-batch
        #     for j in range(0, BATCH_SIZE):
        #         if (y_batch[j][0] == 1):
        #             ytrue_dist_bin = [0, 1]  # true, they are the same genre
        #             ytrue_bin_batch.append(
        #                 ytrue_dist_bin)  # should give a list BATCHSIZE many same_genre boolean targets
        #
        #         elif (y_batch[j][0] == 0):
        #             ytrue_dist_bin = [1, 0]  # false
        #             ytrue_bin_batch.append(
        #                 ytrue_dist_bin)  # should give a list BATCHSIZE many same_genre boolean targets
        #         else:
        #             print("neither true nor false, error, quitting...")
        #             exit(0)
        #
        #     # convert label lists to pytorch tensors
        #     ytrue_bin_batch = np.array(ytrue_bin_batch)
        #     ytrue_bin_batch = torch.from_numpy(ytrue_bin_batch).float()
        #
        #     # passing finetune as the second parameter will skip all the mask token related stuff from pretraining
        #     batch_mask_indices = [[[-1, -1]]]
        #     ypred_bin_batch, ypred_multi_batch = model(X_batch, mask_indices=batch_mask_indices)
        #     ypred_bin_batch = ypred_bin_batch.float()
        #     loss = criterion_bin(ypred_bin_batch, ytrue_bin_batch)
        #     acc = get_accuracy(ypred_bin_batch, ytrue_bin_batch, "inference", None)
        #     epoch_loss += loss.item()
        #     epoch_acc += acc.item()
        #
        #     # get some stats on this batch
        #     for batch_idx, bin_pred in enumerate(
        #             ypred_bin_batch):  # for each 2 dimensional output vector for the binary task
        #         bin_true = ytrue_bin_batch[batch_idx]
        #         true_idx = torch.argmax(bin_true)
        #         pred_idx = torch.argmax(bin_pred)
        #         if (true_idx == pred_idx):
        #             bin_correct_train[true_idx] += 1  # either 0 or 1 was the correct choice, so count it
        #         if (true_idx == 0):
        #             bin_neg_count_train += 1
        #
        #     # update the parameters
        #     # loss.backward()
        #     # optimizer.step()
        #     batch_count += 1

        val_loss = 0
        val_acc = 0
        # running_faw_pos = None
        # running_saw_pos = None
        # running_taw_pos = None
        # running_faw_neg = None
        # running_saw_neg = None
        # running_taw_neg = None
        running_faw = None
        running_saw = None
        running_taw = None
        true_pos_count = 0
        true_neg_count = 0

        for X_batch_val, y_batch_val in val_loader:
            X_batch_val = X_batch_val.float()
            y_batch_val = y_batch_val.float()
            ytrue_bin_batch_val = []  # list of batch targets for binary classification task

            for j in range(0, BATCH_SIZE):
                if (y_batch_val[j][0] == 1):
                    ytrue_dist_bin_val = [0, 1]  # true, they are the same genre
                elif (y_batch_val[j][0] == 0):
                    ytrue_dist_bin_val = [1, 0]  # false
                else:
                    print("label value was not true or false, it was " + str(y_batch_val[j][0]) + ", quitting...")
                    exit(0)
                ytrue_bin_batch_val.append(
                    ytrue_dist_bin_val)  # should give a list BATCHSIZE many same_genre boolean targets

            # convert label lists to pytorch tensors
            ytrue_bin_batch_val = np.array(ytrue_bin_batch_val)
            ytrue_bin_batch_val = torch.from_numpy(ytrue_bin_batch_val).float()

            # passing finetune as the second parameter will skip all the mask token related stuff from pretraining
            #batch_mask_indices = [[[-1, -1]]]
            batch_mask_indices = "finetune"
            ypred_bin_batch_val, ypred_multi_batch_val, faw, saw, taw = model(X_batch_val, mask_indices=batch_mask_indices)


            # print("Faw")
            # print(faw)
            # print("\n")
            # print("Saw")
            # print(saw)
            # print("\n")
            # print("Taw")
            # print(taw)

            # get accuracy stats for validation samples
            for batch_idx, bin_pred in enumerate(
                    ypred_bin_batch_val):  # for each 2 dimensional output vector for the binary task
                bin_true = ytrue_bin_batch_val[batch_idx]
                true_idx = torch.argmax(bin_true)
                pred_idx = torch.argmax(bin_pred)
                if (true_idx == pred_idx):
                    bin_correct_val[true_idx] += 1  # either 0 or 1 was the correct choice, so count it
                if (true_idx == 0):
                    bin_neg_count_val += 1

            if (true_idx == pred_idx): # for now let's only consider the attention weights when the model is correct
                if true_idx == 0: # true negative
                    true_neg_count += 1
                    # if running_faw_neg == None:
                    #     running_faw_neg = faw
                    #     running_saw_neg = saw
                    #     running_taw_neg = taw
                    # else:
                    #     running_faw_neg = torch.add(running_faw_neg, faw)
                    #     running_saw_neg = torch.add(running_saw_neg, saw)
                    #     running_taw_neg = torch.add(running_taw_neg, taw)

                elif true_idx == 1: # true positive
                    true_pos_count += 1
                    # if running_faw_pos == None:
                    #     running_faw_pos = faw
                    #     running_saw_pos = saw
                    #     running_taw_pos = taw
                    # else:
                    #     running_faw_pos = torch.add(running_faw_pos, faw)
                    #     running_saw_pos = torch.add(running_saw_pos, saw)
                    #     running_taw_pos = torch.add(running_taw_pos, taw)
                if running_faw == None:
                    running_faw = faw
                    running_saw = saw
                    running_taw = taw
                else:
                    running_faw = torch.add(running_faw, faw)
                    running_saw = torch.add(running_saw, saw)
                    running_taw = torch.add(running_taw, taw)

            ypred_bin_batch_val = ypred_bin_batch_val.float()
            loss = criterion_bin(ypred_bin_batch_val, ytrue_bin_batch_val)
            acc = get_accuracy(ypred_bin_batch_val, ytrue_bin_batch_val, "inference", None)
            val_loss += loss.item()
            val_acc += acc.item()

    # running_faw_pos_avg = torch.div(running_faw_pos, true_pos_count)
    # running_saw_pos_avg = torch.div(running_saw_pos, true_pos_count)
    # running_taw_pos_avg = torch.div(running_taw_pos, true_pos_count)
    # running_faw_neg_avg = torch.div(running_faw_neg, true_neg_count)
    # running_saw_neg_avg = torch.div(running_saw_neg, true_neg_count)
    # running_taw_neg_avg = torch.div(running_taw_neg, true_neg_count)
    running_faw_avg = torch.div(running_faw, len(val_loader))
    running_saw_avg = torch.div(running_saw, len(val_loader))
    running_taw_avg = torch.div(running_taw, len(val_loader))

    ###### FINALIZE FAW INFORMATION ######
    # faw_head1_pos = running_faw_pos_avg[0][0]
    # faw_head2_pos = running_faw_pos_avg[0][1]
    # faw_head1_neg = running_faw_neg_avg[0][0]
    # faw_head2_neg = running_faw_neg_avg[0][1]
    #
    # print("FAW head1 pos")
    # print(faw_head1_pos)
    # print("FAW head1 neg")
    # print(faw_head1_neg)
    # print("FAW head2 pos")
    # print(faw_head2_pos)
    # print("FAW head2 neg")
    # print(faw_head2_neg)

    faw_head1 = running_faw_avg[0][0]
    faw_head2 = running_faw_avg[0][1]

    print("FAW head1")
    print(faw_head1)
    print("FAW head2")
    print(faw_head2)

    ###### FINALIZE SAW INFORMATION ######
    # saw_head1_pos = running_saw_pos_avg[0][0]
    # saw_head2_pos = running_saw_pos_avg[0][1]
    # saw_head1_neg = running_saw_neg_avg[0][0]
    # saw_head2_neg = running_saw_neg_avg[0][1]
    #
    # print("SAW head1 pos")
    # print(saw_head1_pos)
    # print("SAW head1 neg")
    # print(saw_head1_neg)
    # print("SAW head2 pos")
    # print(saw_head2_pos)
    # print("SAW head2 neg")
    # print(saw_head2_neg)

    saw_head1 = running_saw_avg[0][0]
    saw_head2 = running_saw_avg[0][1]

    print("SAW head1")
    print(saw_head1)
    print("SAW head2")
    print(saw_head2)

    ###### FINALIZE TAW INFORMATION ######
    # taw_head1_pos = running_taw_pos_avg[0][0]
    # taw_head2_pos = running_taw_pos_avg[0][1]
    # taw_head1_neg = running_taw_neg_avg[0][0]
    # taw_head2_neg = running_taw_neg_avg[0][1]
    #
    # print("TAW head1 pos")
    # print(taw_head1_pos)
    # print("TAW head1 neg")
    # print(taw_head1_neg)
    # print("TAW head2 pos")
    # print(taw_head2_pos)
    # print("TAW head2 neg")
    # print(taw_head2_neg)

    taw_head1 = running_taw_avg[0][0]
    taw_head2 = running_taw_avg[0][1]

    print("TAW head1")
    print(taw_head1)
    print("TAW head2")
    print(taw_head2)

    #print("There were "+str(true_pos_count)+" true positives and "+str(true_neg_count)+" true negatives.")
    ##### standard  metrics
    print("Epoch bin val stats:")
    print(f'Epoch {e + 0:03}: | Loss: {val_loss / len(val_loader):.5f} | Acc: {val_acc / len(val_loader):.3f}')
    print("correct counts for this epoch's val split: " + str(bin_correct_val))
    print("bin neg sample count: " + str(bin_neg_count_val))
    print("number of samples: " + str(len(val_loader)))