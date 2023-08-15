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

PosNeg = True # True or False for sametimbre finetuning, were both a True and False sample made for each leftsample?
MASK_INDICES = "finetune" #legacy crap that needs to be replaced, should be finetune or sametimbre
best_listFalsePosNeg = [(0,10), (1,6), (2,10), (3,10), (4,10), (5,10), (6,5), (7,9), (8,10), (9,10)] # (index, epoch) to load weights
best_listTruePosNeg = [(0,9), (1,10), (2,9), (3,10), (4,8), (5,7), (6,10), (7,9), (8,10), (9,10)]
best_listLeftHeldoutSubs = [(0, 5), (1, 7), (2, 4), (3, 6), (4, 9), (5, 4), (6, 9), (7, 8)] # epochs are 0-indexed
best_listRightHeldoutSubs = [(0, 6), (1, 6), (2, 9), (3, 4), (4, 8), (5, 8), (6, 7), (7, 9)]


null_model = False # wildly important that this is set to False when training real models
null_labels  = np.ones(7548) # number of training samples on sametimbre with runs 5-8 held out
null_labels[:3774]=0
np.random.shuffle(null_labels)

debug = 1
val_flag = 1
# this seeding is overwritten at the beginning of main if a count value is passed via command line
device="cpu"
LR_def = 0.00001  # defaults, should normally be set by command line
voxel_dim = 420
src_pad_sequence = [0] * voxel_dim
BATCH_SIZE = 1
orig_dataset = "audimg"

bests_both = [(0, 9), (1, 9), (2, 8), (3, 7), (4, 9), (5, 9), (6, 9), (7, 9), (8, 8), (9, 4), (10, 9), (11, 9)]
bests_CLS_only = [(0, 8), (1, 6), (2, 9), (3, 9), (4, 8), (5, 9), (6, 8), (7, 8), (8, 8), (9, 8), (10, 9), (11, 9)]
print("MASK INDICES IS "+str(MASK_INDICES))
print("orig dataset is "+str(orig_dataset))

if __name__ == "__main__":
    # get command line arguments and options
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    print("Command line opts: " + str(opts))
    print("Command line args: " + str(args))

    if "-m" in opts:
        # -m "this is the description of the run" will be at the end of the command line call
        idx = opts.index("-m")
        run_desc = args[idx]
    else:
        run_desc = None

    # heldout run for finetuning, not the run that was heldout for pretraining
    if "-heldout_run" in opts:
        # the run that was held out in pretraining, which is now held out again, and guides us to the correct saved model
        idx = opts.index("-heldout_run")
        heldout_run = int(args[idx])
        pretrain_idx = args[idx]

    else:
        heldout_run = 0
        pretrain_idx=0
    held_start = (600 + (400 * heldout_run))
    held_range = range(held_start, held_start + 400)


    if "-LR" in opts:
        idx = opts.index("-LR")
        LR = args[idx]
        if LR == "default":
            LR = LR_def  # default value if nothing is passed by command line
        LR = float(LR)
    else:
        LR = LR_def


    # the count in the job submission script's for loop
    if "-iteration" in opts:
        idx = opts.index("-iteration")
        # for samegenre, i want it to hold  out the same data that it held out during pretraining, the index of which is pretrain_idx
        # the variable "thiscount" is only used to find the right dataset, so set that variable to pretrain_idx if we're doing finetuning
        thiscount = int(pretrain_idx)
        iteration = int(args[idx])

        print("this count is " + str(thiscount))
        print("and seed is " + str(args[idx]))
        seed = int(args[idx])
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        thiscount = 0
        iteration = 0

    freeze_pretrained = False
    if "-freeze_pretrained" in opts:
        idx = opts.index("-freeze_pretrained")
        if args[idx] == "True":
            freeze_pretrained = True

    if "-seq_len" in opts:
        idx = opts.index("-seq_len")
        seq_len = int(args[idx])
    else:
        seq_len = 5  # arbitrary default value
    max_length = seq_len*2 + 2 # in the paired models we add 2, one each for CLS and SEP tokens

    # whether or not we want to save the model after training, defaults to False if not provided
    if "-save_model" in opts:
        idx = opts.index("-save_model")
        save_model = args[idx]
        save_model = True if save_model == "True" else False
    else:
        save_model = False

    if "-attention_heads" in opts:
        idx = opts.index("-attention_heads")
        attn_heads = int(args[idx])
    else:
        attn_heads = ATTENTION_HEADS  # defined in Constants.py
    if "-forward_expansion" in opts:
        idx = opts.index("-forward_expansion")
        f_exp = int(args[idx])
    else:
        f_exp = 4  # arbitrary default value

    if "-num_layers" in opts:
        idx = opts.index("-num_layers")
        n_layers = int(args[idx])
    else:
        n_layers = 2  # arbitrary default value

    pretrain_task = "fresh"
    # should be either both or CLS_only or fresh
    if "-pretrain_task" in opts:
        idx = opts.index("-pretrain_task")
        pretrain_task = args[idx]
        if pretrain_task == "both":
            bests_list = bests_both
        elif pretrain_task == "CLS_only":
            bests_list = bests_CLS_only
        elif pretrain_task == "rightHeldoutSubs":
            bests_list = best_listRightHeldoutSubs
        elif pretrain_task == "leftHeldoutSubs":
            bests_list = best_listLeftHeldoutSubs


    if "-finetune_task" in opts:
        idx = opts.index("-finetune_task")
        finetune_task = args[idx]



        if finetune_task == "samegenre":
            dataset = "/isi/music/auditoryimagery2/seanthesis/opengenre/preproc/training_data/2022-11-14/"
            num_subjects = 5
        elif finetune_task == "sametimbre":
            if pretrain_task == "rightHeldoutSubs":
                dataset = "/isi/music/auditoryimagery2/seanthesis/pitchclass/finetuning/sametimbre/datasets/2023-08-03-5TR_audimg_HOruns_hasimagined_1-1-pairs_repetition1_rightSTG_2heldout/"
            elif pretrain_task == "leftHeldoutSubs":
                dataset = "/isi/music/auditoryimagery2/seanthesis/pitchclass/finetuning/sametimbre/datasets/2023-08-03-5TR_audimg_HOruns_hasimagined_1-1-pairs_repetition1_leftSTG_2heldout/"
            elif pretrain_task == "rightHeldoutSubsFresh":
                dataset = "/isi/music/auditoryimagery2/seanthesis/pitchclass/finetuning/sametimbre/datasets/2023-08-03-5TR_audimg_HOruns_hasimagined_1-1-pairs_repetition1_rightSTG_2heldout/"
            elif pretrain_task == "leftHeldoutSubsFresh":
                dataset = "/isi/music/auditoryimagery2/seanthesis/pitchclass/finetuning/sametimbre/datasets/2023-08-03-5TR_audimg_HOruns_hasimagined_1-1-pairs_repetition1_leftSTG_2heldout/"
            else:
            # the 5 at the end is a specific version of the sametimbre dataset, change that last folder for different intentions
                dataset = "/isi/music/auditoryimagery2/seanthesis/pitchclass/finetuning/sametimbre/datasets/2023-04-13-5TR_audimg_HOruns_hasimagined_1-1-pairs_repetition2/"
            num_subjects = 17
        else:
            print("Needs a finetuning task even if it's a fresh model, quitting...")
            quit(0)
    else:
        finetune_task = None

    # when models are saved the filename has a counter on the end so that we don't overwrite previous saves
    saved_model_index = 0
    if "-saved_model_idx" in opts:
        idx = opts.index("-saved_model_idx")
        saved_model_index = str(args[idx])

    # find pretrained states to load if we're not training a fresh model
    if pretrain_task == "fresh" or MASK_INDICES=="sametimbre" or MASK_INDICES=="sametimbre_switch":
        pretrained_model_states = None
    else:
        if finetune_task=="samegenre":
            best_epoch = bests_list[heldout_run][1] # the list at the top of the file tells us which model to load, as the model was saved for every epoch during pretraining
            print("For heldout run {0} we're loading the states saved for {1} pretraining after epoch {2}".format(heldout_run, pretrain_task, best_epoch))
            pretrained_model_states = "/isi/music/auditoryimagery2/seanthesis/thesis/pairedpretrain/trained_models/" + pretrain_task + "/states_" + str(heldout_run) + str(best_epoch)+".pt"
        elif finetune_task=="sametimbre":

            if pretrain_task == "leftHeldoutSubs":
                best_epoch = bests_list[int(saved_model_index)][1] # the list at the top of the file tells us which model to load, as the model was saved for every epoch during pretraining
                saved_model_index = heldout_run
            elif pretrain_task == "rightHeldoutSubs":
                best_epoch = bests_list[int(saved_model_index)][1] # the list at the top of the file tells us which model to load, as the model was saved for every epoch during pretraining
                best_epoch = best_epoch + 10 # filename format stuff, trust
                saved_model_index = heldout_run
            # stuff before thesis revisions
            else:
                if PosNeg == False:
                    best_epoch = best_listFalsePosNeg[int(saved_model_index)][1] # the list at the top of the file tells us which model to load, as the model was saved for every epoch during pretraining
                    best_epoch = best_epoch - 1 # the bests list runs from 1 to 10, saved weights go from 0 to 9
                elif PosNeg == True:
                    best_epoch = best_listTruePosNeg[int(saved_model_index)][1] # the list at the top of the file tells us which model to load, as the model was saved for every epoch during pretraining
                    best_epoch = best_epoch - 1 + 10# the bests list runs from 1 to 10, saved weights go from 0 to 9
            print("For heldout run {0} we're loading the states saved for {1} pretraining after epoch {2}".format(heldout_run, pretrain_task, best_epoch))
            pretrained_model_states = "/isi/music/auditoryimagery2/seanthesis/thesis/pairedpretrain/trained_models/audimgrevision/CLS_only/states_{0}{1}.pt".format(saved_model_index, best_epoch)
            # pretrained_model_states = "/isi/music/auditoryimagery2/seanthesis/thesis/pairedpretrain/trained_models/audimg/CLS_only/states_09.pt"



    today = datetime.date.today()
    now = datetime.datetime.now()
    ##############################  SET PARAMETERS  ##############################
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dictionary of hyperparameters, eventually should probably come from command line
    hp_dict = {

        "pretrain_task": pretrain_task,  # throughout the code, task is what the model was pretrained on
        # CLS and MSK task are also referring to what the model was trained on
        "CLS_task": "sametimbre",  # samegenre or sametimbre
        "num_CLS_labels": 2,
        "MSK_task": "finetune",
        "finetune_task": finetune_task,
        "COOL_DIVIDEND": COOL_DIVIDEND,
        "ATTENTION_HEADS": attn_heads,
        "num_layers": n_layers,
        "device": str(device),
        "MSK_flag": 1,
        "CLS_flag": 1,
        "BATCH_SIZE": 1,
        "EPOCHS": FINETUNE_EPOCHS, # defined in Constants.py
        "LEARNING_RATE": LR,  # set at top of this file or by command line argument
        # Have to manually set the name of the folder whose training data you want to use, since there will be many
        "data_dir": dataset,
        # Manually set the hemisphere and iteration number of the dataset you want to use in that folder
        "hemisphere": "left",
        "count": str(heldout_run),
        "iteration": iteration,
        # manually set max_seq_length used in data creation, in the input CLS+seq+SEP+seq this is the max length of seq
        "max_sample_length": seq_len,
        "within_subject": 1,
        "num_subjects": num_subjects,
        "heldout_run": heldout_run,
        "held_start": held_start,  # first 600 indices of refsamples are from testruns
        "held_range": held_range,
        "forward_expansion": f_exp
    }

    hp_dict["data_path"] = hp_dict["data_dir"]
    torch.set_default_dtype(torch.float32)

    # set up logfile, PRETRAIN_LOG_PATH is defined in Constants.py
    today_dir = THESIS_PATH + "pairedfinetune/logs/" + str(today) + "/"
    if not (os.path.exists(today_dir)):
        os.mkdir(today_dir)

    if (thiscount != None):
        logcount = thiscount
    else:
        logcount = 0
    logfile = today_dir + "pairedfinetunelog_" + str(logcount) + ".txt"
    while (os.path.exists(logfile)):
        logcount += 1
        logfile = today_dir + "pretrainlog_" + str(logcount) + ".txt"
    log = open(logfile, "w")
    log.write(str(now) + "\n")

    # run_desc is potentially given in command line call
    if (run_desc is not None):
        log.write(run_desc + "\n\n")
        print(run_desc + "\n\n")
    # write hyperparameters to log
    for hp in hp_dict.keys():
        log.write(str(hp) + " : " + str(hp_dict[hp]) + "\n")

    # load the training and validation data
    if orig_dataset == "genre":
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_samples" + hp_dict["count"] + ".p",
                  "rb") as samples_fp:
            train_X = pickle.load(samples_fp)
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_labels" + hp_dict["count"] + ".p",
                  "rb") as labels_fp:
            train_Y_detailed = pickle.load(labels_fp)

    elif orig_dataset == "audimg":
        with open(hp_dict["data_path"] + "all_X_fold"+str(heldout_run)+".p", "rb") as samples_fp:
            train_X = pickle.load(samples_fp)
        with open(hp_dict["data_path"] + "all_y_fold"+str(heldout_run)+".p", "rb") as labels_fp:
            train_Y_detailed = pickle.load(labels_fp)


    if orig_dataset == "genre":
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_valsamples" + hp_dict["count"] + ".p",
                  "rb") as samples_fp:
            val_X = pickle.load(samples_fp)
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_vallabels" + hp_dict["count"] + ".p",
                  "rb") as labels_fp:
            val_Y_detailed = pickle.load(labels_fp)
    elif orig_dataset == "audimg":
        with open(hp_dict["data_path"] + "all_val_X_fold"+str(heldout_run)+".p", "rb") as samples_fp:
            val_X = pickle.load(samples_fp)
        with open(hp_dict["data_path"] + "all_val_y_fold"+str(heldout_run)+".p", "rb") as labels_fp:
            val_Y_detailed = pickle.load(labels_fp)

    # the labels are detailed tuples, so let's just extract the True/False value into a list that we can turn into a tensor
    train_y = []
    val_y = []

    for detailed_label in train_Y_detailed:
        tf = detailed_label[0]  # True/False is the first thing in the tuple
        if tf == 1:
            train_y.append([1])
        elif tf == 0:
            train_y.append([0])
        else:
            print("label value neither True nor False in detailed label " + str(detailed_label) + ", quitting...")
            exit(0)

    for detailed_val_label in val_Y_detailed:
        tf = detailed_val_label[0]  # True/False is the first thing in the tuple
        if tf == 1:
            val_y.append([1])
        elif tf == 0:
            val_y.append([0])
        else:
            print(
                "label value neither True nor False in detailed val label " + str(detailed_val_label) + ", quitting...")
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
    model = Transformer(num_CLS_labels=hp_dict["num_CLS_labels"], num_genres=10, src_pad_sequence=src_pad_sequence,
                        max_length=max_length, voxel_dim=voxel_dim, ref_samples=None, mask_task=hp_dict["MSK_task"], print_flag=0,
                        heads=hp_dict["ATTENTION_HEADS"], num_layers=hp_dict["num_layers"],
                        forward_expansion=hp_dict["forward_expansion"]).to(hp_dict["device"])

    # if we want to load pretrained weights:
    if pretrain_task not in ["fresh", "leftHeldoutSubsFresh", "rightHeldoutSubsFresh"]:
        model.load_state_dict(torch.load(pretrained_model_states), strict=False)
        if freeze_pretrained:
            finetune_params = ["output_layer_finetune.1.bias", "output_layer_finetune.1.weight",
                               "output_layer_finetune.0.bias", "output_layer_finetune.0.weight"]

            params = model.state_dict()
            training_tensors = (params[tensor_name] for tensor_name in finetune_params)
        else:
            training_tensors = model.parameters()
    else:
        training_tensors = model.parameters()
    model = model.float()

    criterion_bin = nn.CrossEntropyLoss()
    # training_tensors is either everything or only the new layers for finetuning when freeze_pretrained is True
    optimizer = optim.Adam(training_tensors, lr=hp_dict["LEARNING_RATE"], betas=(0.9, 0.999), weight_decay=0.0001)

    # train
    best_avg_val_acc = 0
    best_val_epoch = -1
    for e in range(1, hp_dict["EPOCHS"]+1):
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
        for X_batch, y_batch in train_loader:
            if null_model:
                null_label = null_labels[batch_count]
                y_batch = [[null_label]]
                y_batch = np.array(y_batch)
                y_batch = torch.from_numpy(y_batch)
            X_batch = X_batch.float()
            y_batch = y_batch.float()
            # X_batch, y_batch = X_batch.to(hp_dict["device"]), y_batch.to(hp_dict["device"])
            ytrue_bin_batch = []  # list of batch targets for binary classification task

            optimizer.zero_grad()  # reset gradient to zero before each mini-batch
            for j in range(0, BATCH_SIZE):
                if (y_batch[j][0] == 1):
                    ytrue_dist_bin = [0, 1]  # true, they are the same genre
                    ytrue_bin_batch.append(
                        ytrue_dist_bin)  # should give a list BATCHSIZE many same_genre boolean targets

                elif (y_batch[j][0] == 0):
                    ytrue_dist_bin = [1, 0]  # false
                    ytrue_bin_batch.append(
                        ytrue_dist_bin)  # should give a list BATCHSIZE many same_genre boolean targets
                else:
                    print("neither true nor false, error, quitting...")
                    exit(0)

            # convert label lists to pytorch tensors
            ytrue_bin_batch = np.array(ytrue_bin_batch)
            ytrue_bin_batch = torch.from_numpy(ytrue_bin_batch).float()

            # passing finetune as the second parameter will skip all the mask token related stuff from pretraining
            ypred_bin_batch = model(X_batch, mask_indices=MASK_INDICES)
            ypred_bin_batch = ypred_bin_batch.float()
            loss = criterion_bin(ypred_bin_batch, ytrue_bin_batch)
            acc = get_accuracy(ypred_bin_batch, ytrue_bin_batch, hp_dict["finetune_task"], None)
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
            batch_count += 1

        # after all training batches for this epoch
        # now the validation split
        with torch.no_grad():
            # change model to evaluation mode
            model.eval()
            val_loss = 0
            val_acc = 0
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
                ypred_bin_batch_val = model(X_batch_val, mask_indices=MASK_INDICES)

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
                acc = get_accuracy(ypred_bin_batch_val, ytrue_bin_batch_val, hp_dict["finetune_task"], None)
                val_loss += loss.item()
                val_acc += acc.item()

            # after batch
        # after with torch.no_grad()
        # print training stats for this epoch
        avg_val_acc = val_acc / len(val_loader)

        print("Epoch bin training stats:")
        print(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
        print("correct counts for this epoch: " + str(bin_correct_train))
        print("bin neg sample count: " + str(bin_neg_count_train))
        print("number of samples: " + str(len(train_loader)))
        print("\n")
        print("Epoch bin val stats:")
        print(f'Epoch {e + 0:03}: | Loss: {val_loss / len(val_loader):.5f} | Acc: {val_acc / len(val_loader):.3f}')
        print("correct counts for this epoch's val split: " + str(bin_correct_val))
        print("bin neg sample count: " + str(bin_neg_count_val))
        print("number of samples: " + str(len(val_loader)))

        if save_model: # save model is set by command line argument
            if avg_val_acc > best_avg_val_acc:
                #modelcount=0
                best_avg_val_acc = avg_val_acc
                best_val_epoch = e
                model_path = "/isi/music/auditoryimagery2/seanthesis/thesis/pairedfinetune/trained_models/"+str(hp_dict["finetune_task"])+"/"+str(hp_dict["pretrain_task"])+"/states_"+str(thiscount)+".pt"


                torch.save(model.state_dict(),model_path)
                model_path = "/isi/music/auditoryimagery2/seanthesis/thesis/pairedfinetune/trained_models/"+str(hp_dict["finetune_task"])+"/"+str(hp_dict["pretrain_task"])+"/full_" + str(thiscount) + ".pt"
                torch.save(model,model_path)
    print("Best val epoch was "+str(best_val_epoch))

    log.close()

