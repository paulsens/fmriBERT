# finetune a transfer transformer on the same-timbre task
# started as a copy of pretrain.py with adjustments for pure evaluation

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

debug=1
val_flag=1
# this seeding is overwritten at the beginning of main if a count value is passed via command line

LR_def=0.0001 #defaults, should normally be set by command line
voxel_dim=420
src_pad_sequence = [0] * voxel_dim
FINETUNE_EPOCHS=10
BATCH_SIZE=1

if __name__=="__main__":
    # get command line arguments and options
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if "-m" in opts:
        # -m "this is the description of the run" will be at the end of the command line call
        idx= opts.index("-m")
        run_desc = args[idx]
    else:
        run_desc = None
    print("Command line opts: "+str(opts))
    print("Command line args: "+str(args))

    if "-pretrain_task" in opts:
        idx = opts.index("-pretrain_task")
        # this value will be "fresh" rather than a number if we want to train a fresh model
        pretrain_task = args[idx]

    else:
        print("provide the task the desired pretrained weights were trained on, i.e \"-pretrain_task both\" or \"-pretrain_idx fresh\" if training a fresh model. Choices are both, binaryonly, multionly, fresh")
        quit(0)

    # the index assigned to the loaded model in pretraining
    if "-pretrain_idx" in opts:
        idx = opts.index("-pretrain_idx")
        # this value will be "fresh" rather than a number if we want to train a fresh model
        # can probably keep this as a string
        pretrain_idx = args[idx]

        # this is how we'll initialize the RNG seeds
        #seed = 20 if pretrain_idx=="fresh" else int(pretrain_idx)
        #random.seed(seed)
        #torch.manual_seed(seed)
        #np.random.seed(seed)
    else:
        print("assign a pretrain model index to load, i.e \"-pretrain_idx 5\" or \"-pretrain_idx fresh\" if training a fresh model. choices are 0 to 11 inclusive")
        quit(0)

    # the count in the job submission script's for loop
    if "-count" in opts:
        idx = opts.index("-count")
        thiscount = int(args[idx])
        print("this count is "+str(thiscount))
        seed=thiscount
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        thiscount = None

    freeze_pretrained = False
    if "-freeze_pretrained" in opts:
        idx = opts.index("-freeze_pretrained")
        if args[idx]=="True":
            freeze_pretrained=True

    # env is defined in Constants.py
    if env=="local":
        pretrained_model_states = "/Volumes/External/opengenre/official/"+pretrain_task+"/states_"+pretrain_idx+".pt"
        data_path = "/Volumes/External/pitchclass/finetuning/sametimbre/datasets/2/"
    if env=="discovery":
        pretrained_model_states = "/isi/music/auditoryimagery2/seanthesis/opengenre/final/"+pretrain_task+"/states_"+pretrain_idx+"0.pt"
        data_path = "/isi/music/auditoryimagery2/seanthesis/pitchclass/finetuning/sametimbre/datasets/5/"

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
    model = Transformer(next_sequence_labels=2, num_genres=10, src_pad_sequence=src_pad_sequence, max_length=12, voxel_dim=voxel_dim, ref_samples=None, mask_task=None, print_flag=0)


    # if we want to load pretrained weights:
    if pretrain_task != "fresh":
        model.load_state_dict(torch.load(pretrained_model_states), strict=False)
        if freeze_pretrained:
            finetune_params = ["output_layer_finetune.1.bias", "output_layer_finetune.1.weight", "output_layer_finetune.0.bias", "output_layer_finetune.0.weight"]

            params = model.state_dict()
            training_tensors = (params[tensor_name] for tensor_name in finetune_params)
        else:
            training_tensors = model.parameters()
    else:
        training_tensors = model.parameters()
    model = model.float()

    criterion_bin = nn.CrossEntropyLoss()
    # training_tensors is either everything or only the new layers for finetuning when freeze_pretrained is True
    optimizer = optim.Adam(training_tensors, lr=LR_def, betas=(0.9, 0.999), weight_decay=0.0001)

    # train
    for e in range(0, FINETUNE_EPOCHS):
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

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.float()
            y_batch = y_batch.float()
            #X_batch, y_batch = X_batch.to(hp_dict["device"]), y_batch.to(hp_dict["device"])
            ytrue_bin_batch = []  # list of batch targets for binary classification task

            optimizer.zero_grad()  # reset gradient to zero before each mini-batch
            for j in range(0, BATCH_SIZE):
                if (y_batch[j][0]==1):
                    ytrue_dist_bin = [0, 1]  # true, they are the same genre
                    ytrue_bin_batch.append(ytrue_dist_bin)  # should give a list BATCHSIZE many same_genre boolean targets

                elif(y_batch[j][0]==0):
                    ytrue_dist_bin = [1, 0]  # false
                    ytrue_bin_batch.append(ytrue_dist_bin)  # should give a list BATCHSIZE many same_genre boolean targets
                else:
                    print("neither true nor false, error, quitting...")
                    exit(0)

            # convert label lists to pytorch tensors
            ytrue_bin_batch = np.array(ytrue_bin_batch)
            ytrue_bin_batch = torch.from_numpy(ytrue_bin_batch).float()

            # passing finetune as the second parameter will skip all the mask token related stuff from pretraining
            ypred_bin_batch = model(X_batch, mask_indices="finetune")
            ypred_bin_batch = ypred_bin_batch.float()
            loss = criterion_bin(ypred_bin_batch, ytrue_bin_batch)
            acc = get_accuracy(ypred_bin_batch, ytrue_bin_batch, "sametimbre", None)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

            # get some stats on this batch
            for batch_idx, bin_pred in enumerate(
                    ypred_bin_batch):  # for each 2 dimensional output vector for the binary task
                bin_true = ytrue_bin_batch[batch_idx]
                true_idx = torch.argmax(bin_true)
                pred_idx = torch.argmax(bin_pred)
                if (true_idx == pred_idx):
                    bin_correct_train[true_idx] += 1  # either 0 or 1 was the correct choice, so count it
                if (true_idx == 0):
                    bin_neg_count_train += 1

            # update the parameters
            loss.backward()
            optimizer.step()

        # after all training batches for this epoch
        # now the validation split
        with torch.no_grad():
            # change model to evaluation mode
            model.eval()
            val_loss=0
            val_acc=0
            for X_batch_val, y_batch_val in val_loader:
                X_batch_val = X_batch_val.float()
                y_batch_val = y_batch_val.float()
                ytrue_bin_batch_val = []  # list of batch targets for binary classification task

                for j in range(0, BATCH_SIZE):
                    if (y_batch_val[j][0]==1):
                        ytrue_dist_bin_val = [0, 1]  # true, they are the same genre
                    elif(y_batch_val[j][0]==0):
                        ytrue_dist_bin_val = [1, 0]  # false
                    else:
                        print("label value was not true or false, it was "+str(y_batch_val[j][0])+", quitting...")
                        exit(0)
                    ytrue_bin_batch_val.append(ytrue_dist_bin_val)  # should give a list BATCHSIZE many same_genre boolean targets

                # convert label lists to pytorch tensors
                ytrue_bin_batch_val = np.array(ytrue_bin_batch_val)
                ytrue_bin_batch_val = torch.from_numpy(ytrue_bin_batch_val).float()

                # passing finetune as the second parameter will skip all the mask token related stuff from pretraining
                ypred_bin_batch_val= model(X_batch_val, mask_indices="finetune")

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

                ypred_bin_batch_val = ypred_bin_batch_val.float()
                loss = criterion_bin(ypred_bin_batch_val, ytrue_bin_batch_val)
                acc = get_accuracy(ypred_bin_batch_val, ytrue_bin_batch_val, "sametimbre", None)
                val_loss += loss.item()
                val_acc += acc.item()

        # end epoch



