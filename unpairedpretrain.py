import torch
import torch.nn as nn
import random
from random import randint
import numpy as np
from helpers import *
from transfer_transformer import *
from pitchclass_data import *
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from Constants import *
import sys
import os
import datetime

debug = 1
val_flag = 1
seed = 3
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
mask_variation = False
CLS_task_weight = 0.5
MSK_task_weight = 1 - CLS_task_weight
LR_def = 0.0001  # defaults, should normally be set by command line
printed_count = 0
val_printed_count = 0
# torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        # get command line arguments and options
        opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
        args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

        # text description of this job
        if "-m" in opts:
            # -m "this is the description of the run" will be at the end of the command line call
            idx = opts.index("-m")
            run_desc = args[idx]
        else:
            run_desc = None

        # pretraining task
        if "-task" in opts:
            # either "both", "CLS_only" or "MSK_only"
            idx = opts.index("-task")
            task_arg = args[idx]
        else:
            task_arg = "both"

        # train on gpu or not, not implemented yet
        if "-gpu" in opts:
            idx = opts.index("-gpu")
            gpunum = args[idx]  # currently only works if only one gpu is given
            device = torch.device("cuda:" + str(gpunum))
        else:
            device = "cpu"

        # index in job submission script, indicates the fold of the dataset to train on, not necessarily a held out run
        if "-training_fold" in opts:
            idx = opts.index("-training_fold")
            training_fold = int(args[idx])
        else:
            training_fold = 0


        if "-LR" in opts:
            idx = opts.index("-LR")
            LR = args[idx]
            if LR == "default":
                LR = LR_def  # default value if nothing is passed by command line
            LR = float(LR)
        else:
            LR = 0.00001

        if "-CLS_task_weight" in opts:
            idx = opts.index("-CLS_task_weight")
            CLS_task_weight = args[idx]
            if CLS_task_weight == "default":
                CLS_task_weight = 0.5
            else:
                CLS_task_weight = float(CLS_task_weight)
            MSK_task_weight = 1 - CLS_task_weight  # if we're just averaging then these two weights would both be 0.5
        else:
            CLS_task_weight = 0.5
            MSK_task_weight = 1 - CLS_task_weight
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

        if "-dataset" in opts:
            idx = opts.index("-dataset")
            dataset = str(args[idx]) # either genre or audimg
        else:
            dataset = "genre"  # arbitrary default value
        if dataset=="audimg":
            datadir = "2023-03-27-10TR_audimg_HOruns_hasimagined"
            # labels made from audimg dataset look like   (timbre, cond, subid, run_n, cycle_n, majorkey, stimnote, stimoctave, vividness, GoF, choice)
            dir_idx = 10
            num_subjects = 17
            training_fold = ""

        elif dataset=="genre":
            datadir = "2023-03-27-10TR_genre_HOruns"
            # labels made from genre dataset look like [direction, this_genre, i, sub_id_int]
            dir_idx = 0 # how to find the direction in the label
            num_subjects = 5
        else:
            print("illegal dataset, should be either audimg or genre, quitting...")
            quit(0)

        today = datetime.date.today()
        now = datetime.datetime.now()
        ##############################  SET PARAMETERS  ##############################
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dictionary of hyperparameters, eventually should probably come from command line
        hp_dict = {

            "task": task_arg,  # passed in by command line, both or CLS_only
            "dataset":dataset, # either audimg or genre
            "CLS_task": "timedir",  # same_genre or nextseq
            "num_CLS_labels": 2,
            "MSK_task": "reconstruction", # aka Masked Brain Modeling
            "COOL_DIVIDEND": COOL_DIVIDEND,
            "ATTENTION_HEADS": attn_heads,
            "forward_expansion": f_exp,
            "num_layers": n_layers,
            "device": str(device),
            "MSK_flag": 1,
            "CLS_flag": 1,
            "BATCH_SIZE": 1,
            "EPOCHS": EPOCHS,
            "LEARNING_RATE": LR,  # set at top of this file or by command line argument
            # Have to manually set the name of the folder whose training data you want to use, since there will be many
            "data_dir": datadir,  # set in arg parsing on -dataset
            # Manually set the hemisphere and iteration number of the dataset you want to use in that folder
            "hemisphere": "left",
            "training_fold": str(training_fold),
            "max_sample_length": 10,  # manually set max_seq_length used in data creation, does not include CLS token

            "mask_variation": mask_variation,
            # whether to just pick one thing to mask at random, or 50% of picking two things
            # BERT uses the latter, but when sample length is only 5 i think two masks is too difficult for the model
            "within_subject": 1,
            # this doesn't really do anything now that inputs aren't paired, but it does serve as a reminder that the training data was created within subject (or not, potentially in the future)
            "num_subjects": num_subjects,  # nakai et al's music genre dataset had 5 subjects, michael's had 17
        }

        if (debug):
            print("LR is " + str(hp_dict["LEARNING_RATE"]))
            print("CLS_task_weight is " + str(CLS_task_weight))
            print("MSK_task_weight is " + str(MSK_task_weight))

        # SET THE DATAPATH
        hp_dict["data_path"] = timedir_datasets_path + hp_dict["data_dir"] + "/"
        torch.set_default_dtype(torch.float32)

        # set up logfile, PRETRAIN_LOG_PATH is defined in Constants.py
        # task is either both, CLS_only or MSK_only
        today_dir = THESIS_PATH+"unpairedpretrain/logs/"+str(today)+"/"
        if not (os.path.exists(today_dir)):
            os.mkdir(today_dir)
        if (training_fold != None):
            # include the index of the heldout run, if there is one, to differentiate logs
            logcount = training_fold
        else:
            logcount = 0
        logfile = today_dir + "pretrainlog_" + str(logcount) + ".txt"

        log = open(logfile, "w")
        log.write(str(now) + "\n")
        # run_desc is potentially given in command line call
        if (run_desc is not None):
            log.write(run_desc + "\n\n")
            print(run_desc + "\n\n")
        # write hyperparameters to log and stdout
        for hp in hp_dict.keys():
            log.write(str(hp) + " : " + str(hp_dict[hp]) + "\n")
        print("HP dict: "+str(hp_dict))

        # load samples and labels created with a particular run heldout from all subjects
        # if no heldout run was given, it defaults to the data created with run 0 held out, i.e hp_dict["count"] is 0
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_samples" + hp_dict["training_fold"] + ".p",
                  "rb") as samples_fp:
            train_X = pickle.load(samples_fp)
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_labels" + hp_dict["training_fold"] + ".p",
                  "rb") as labels_fp:
            train_Y = pickle.load(labels_fp)
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_refsamples.p", "rb") as refsamples_fp:
            ref_samples = pickle.load(refsamples_fp)

        # load valsamples and vallabels
        if (val_flag == 0):
            val_X = None
            val_Y = None
        else:
            # load the validation data created from the heldout run (from all subjects)
            with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_valsamples" + hp_dict["training_fold"] + ".p",
                      "rb") as samples_fp:
                val_X = pickle.load(samples_fp)
            with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_vallabels" + hp_dict["training_fold"] + ".p",
                      "rb") as labels_fp:
                val_Y = pickle.load(labels_fp)
            print("num val samples is " + str(len(val_X)))
            # print("val Y is "+str(val_Y))

        # train_X has shape (timesteps, max_length, voxel_dim)
        num_samples = len(train_X)
        max_length = len(train_X[0])  # should be seq_len + 1
        print("max length is " + str(max_length) + " and hpdict is " + str(hp_dict["max_sample_length"]))
        # assert (max_length == (hp_dict["max_sample_length"] + 1))
        voxel_dim = len(train_X[0][0])
        print("voxel dim is " + str(voxel_dim))
        print("num samples is " + str(num_samples))
        # print("train Y is "+str(train_Y))

        # convert to numpy arrays
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
        print("train x has shape " + str(train_X.shape))
        if val_X is not None:
            val_X = np.array(val_X)
            val_Y = np.array(val_Y)

        # convert to tensors
        train_X = torch.from_numpy(train_X)
        train_Y = torch.from_numpy(train_Y)
        if val_X is not None:
            val_X = torch.from_numpy(val_X)
            val_Y = torch.from_numpy(val_Y)

        # make the TrainData object
        all_data = TrainData(train_X, train_Y)
        if val_X is not None:
            val_data = TrainData(val_X, val_Y)  # make TrainData object for validation data

        # make DataLoader objects
        train_loader = DataLoader(dataset=all_data, batch_size=hp_dict["BATCH_SIZE"],
                                  shuffle=True)  # make the DataLoader object
        if val_X is not None:
            val_loader = DataLoader(dataset=val_data, batch_size=hp_dict["BATCH_SIZE"], shuffle=False)

        MSK_token = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
        MSK_token = np.array(MSK_token)
        MSK_token = torch.from_numpy(MSK_token)

        CLS_task_labels = 2  # two possible labels for same genre task, yes or no
        num_genres = 10  # from the training set, not used currently

        src_pad_sequence = [0] * voxel_dim  # not used currently

        model = Transformer(num_CLS_labels=hp_dict["num_CLS_labels"], num_genres=num_genres,
                            src_pad_sequence=src_pad_sequence, max_length=max_length, voxel_dim=voxel_dim,
                            ref_samples=ref_samples, mask_task=hp_dict["MSK_task"], print_flag=0,
                            heads=hp_dict["ATTENTION_HEADS"], num_layers=hp_dict["num_layers"],
                            forward_expansion=hp_dict["forward_expansion"]).to(hp_dict["device"])
        model = model.float()
        model.to(hp_dict["device"])

        criterion_bin = nn.CrossEntropyLoss()
        if hp_dict["MSK_task"] == "genre_decode":
            criterion_multi = nn.CrossEntropyLoss()
            get_multi_acc = True
        elif hp_dict["MSK_task"] == "reconstruction":
            criterion_multi = nn.MSELoss()
            get_multi_acc = False
        optimizer = optim.Adam(model.parameters(), lr=hp_dict["LEARNING_RATE"], betas=(0.9, 0.999), weight_decay=0.0001)

        # LOOP OVER EPOCHS
        for e in range(1, hp_dict["EPOCHS"] + 1):
            batch_i = 0

            epoch_val_masks = []
            bin_correct_train = [0,
                                 0]  # 0'th index is the number of times the model was correct when the ground truth was 0, and when ground truth was 1

            bin_correct_val = [0, 0]
            bin_neg_count_train = 0  # count the number of training samples where 0 was the correct answer
            bin_neg_count_val = 0  # count the number of validation samples where 0 was the correct answer

            # reproducible seeds based on epoch number
            random.seed(seed + e)
            torch.manual_seed(seed + e)
            np.random.seed(seed + e)

            model.train()  # sets model status, doesn't actually start training
            # needed every epoch because the validation part below sets model.eval()
            epoch_loss = 0  # weighted average of two tasks' losses
            epoch_acc = 0  # accuracy for CLS task
            epoch_acc2 = 0  # accuracy for MSK task
            epoch_tbin_loss = 0  # loss for CLS task on training split
            epoch_vbin_loss = 0  # loss for CLS task on validation split (just a metric, isn't backpropagated obv)
            epoch_tmulti_loss = 0  # analogous for MSK task
            epoch_vmulti_loss = 0

            # LOOP OVER BATCHES
            for X_batch, y_batch in train_loader:
                batch_mask_indices = []  # a list of BATCH_SIZE many lists. Element n is a list of two ints. Those two ints are the indices of sample n which were masked. If only one index was masked, the second int will be -1, and the model will react appropriately downstream. For example, [1, 5] means that timesteps 1 and 5 in the input sequence were masked (note it will never contain 0 as that is the CLS token). This list is required for the model to know which element(s) of the transformed sequence to send to the output layer for reconstruction
                X_batch = X_batch.float()  # inconsistencies between doubles and floats naturally, so force everything to float
                y_batch = y_batch.float()
                X_batch, y_batch = X_batch.to(hp_dict["device"]), y_batch.to(hp_dict["device"])
                ytrue_bin_batch = []  # list of batch targets for CLS task
                ytrue_multi_batch = []  # list of batch targets for MSK task

                # LOOP OVER SAMPLES IN BATCH
                optimizer.zero_grad()  # reset gradient to zero before each mini-batch
                for x in range(0, hp_dict["BATCH_SIZE"]):
                    sample_mask_indices = []  # will contain two ints, see comment for batch_mask_indices above
                    #print("before masking, X_batch is "+str(X_batch))
                    # no return value from apply_masks, everything is updated by reference in the lists
                    apply_masks_timedir(X_batch[x], y_batch[x], ref_samples, hp_dict, mask_variation, ytrue_multi_batch,
                                        batch_mask_indices, sample_mask_indices, mask_task=hp_dict["MSK_task"], log=log,
                                        heldout=False, task=hp_dict["task"])

                    # print("after masking, X_batch is "+str(X_batch))
                    # print("batch mask indices is "+str(batch_mask_indices))
                    # print("sample mask indices is "+str(sample_mask_indices))
                    # print("ytrue multi batch is "+str(ytrue_multi_batch))
                    # quit(0)

                for y in range(0, hp_dict["BATCH_SIZE"]):
                    if (y_batch[y][dir_idx] == 1):  # then it WAS reversed
                        ytrue_dist_bin = [0, 1]  # a one-hot probdist for "yes it was reversed"
                    else:
                        ytrue_dist_bin = [1, 0]  # a one-hot probdist for "no it was not reversed"
                    ytrue_bin_batch.append(
                        ytrue_dist_bin)  # should give a list BATCHSIZE many same_genre boolean targets

                # convert label lists to pytorch tensors
                ytrue_bin_batch = np.array(ytrue_bin_batch)
                ytrue_multi_batch = np.array(ytrue_multi_batch)

                # batch_mask_indices = np.array(batch_mask_indices)

                ytrue_bin_batch = torch.from_numpy(ytrue_bin_batch).float()
                ytrue_multi_batch = torch.from_numpy(ytrue_multi_batch).float()

                # batch_mask_indices = torch.from_numpy(batch_mask_indices).float()

                # send this stuff to device
                ytrue_bin_batch.to(hp_dict["device"])
                # batch_mask_indices.to(hp_dict["device"])

                # returns predictions for CLS task and MSK task in that order
                if batch_i == 0:
                    print("X batch: "+str(X_batch))

                ypred_bin_batch, ypred_multi_batch = model(X_batch, batch_mask_indices)
                if batch_i == 0:
                    print("ypred bin batch: "+str(ypred_bin_batch))

                for batch_idx, bin_pred in enumerate(
                        ypred_bin_batch):  # for each 2 dimensional output vector for the CLS task
                    bin_true = ytrue_bin_batch[batch_idx]
                    true_idx = torch.argmax(bin_true)
                    pred_idx = torch.argmax(bin_pred)

                    # these two variables are just for bookkeeping/metadata
                    if (true_idx == pred_idx):
                        bin_correct_train[true_idx] += 1  # either 0 or 1 was the correct choice, so count it
                    if (true_idx == 0):
                        bin_neg_count_train += 1  # how many negative samples were correct? will subtract from total correct later to get positive samples correct, then we can analyze.

                ypred_bin_batch = ypred_bin_batch.float()
                ypred_multi_batch = ypred_multi_batch.float()

                # print("attempted to reconstruct "+str(ytrue_multi_batch))
                loss_bin = criterion_bin(ypred_bin_batch, ytrue_bin_batch)
                loss_multi = criterion_multi(ypred_multi_batch, ytrue_multi_batch)

                # the total loss of this sample depends on whether we're doing simultaneous training or not
                if hp_dict["task"] == "CLS_only":
                    loss = loss_bin
                    if batch_i == 0:
                        print("loss_bin: " + str(loss_bin))
                    acc = get_accuracy(ypred_bin_batch, ytrue_bin_batch, hp_dict["CLS_task"], log)
                elif hp_dict["task"] == "MSK_only":
                    loss = loss_multi
                    acc = get_accuracy(ypred_multi_batch, ytrue_multi_batch, hp_dict["MSK_task"], log)

                elif hp_dict["task"] == "both":
                    loss = (loss_bin * CLS_task_weight) + (
                                loss_multi * MSK_task_weight)  # as per devlin et al, loss is a weighted average of the two tasks' losses, the weights are tuned hyperparameters although devlin did 0.5 each
                    if batch_i == 0:
                        print("loss_bin: " + str(loss_bin))
                        print("loss: "+str(loss))
                    epoch_tbin_loss += loss_bin.item()  # keeping track of CLS and MSK losses separately for analysis
                    epoch_tmulti_loss += loss_multi.item()
                    acc1 = get_accuracy(ypred_bin_batch, ytrue_bin_batch, hp_dict["CLS_task"], log)
                    acc2 = get_accuracy(ypred_multi_batch, ytrue_multi_batch, hp_dict["MSK_task"], log)

                if batch_i == 0:
                    print("loss before backward: "+str(loss))

                loss.backward()
                optimizer.step()
                batch_i += 1
                # update stats for this epoch
                epoch_loss += loss.item()
                if (hp_dict["task"] == "both"):
                    epoch_acc += acc1.item()
                    epoch_acc2 += acc2.item()
                else:
                    epoch_acc += acc.item()
                    epoch_acc2 += 0

            # now calculate validation loss/acc, turn off gradient
            if val_X is not None:
                model.eval()

                # the validation split isn't shuffled so this seeding shouldn't do anything but just for posterity...
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)

                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0
                    val_acc2 = 0

                    # LOOP THROUGH BATCHES
                    for X_batch_val, y_batch_val in val_loader:
                        X_batch_val = X_batch_val.float()
                        y_batch_val = y_batch_val.float()
                        batch_mask_indices_val = []  # see comment above for batch_mask_indices
                        ytrue_bin_batch_val = []  # list of batch targets for CLS task
                        ytrue_multi_batch_val = []  # list of batch targets for MSK task
                        X_batch_val, y_batch_val = X_batch_val.to(hp_dict["device"]), y_batch_val.to(hp_dict["device"])

                        # LOOP THROUGH SAMPLES WITHIN THIS BATCH
                        for x in range(0, hp_dict["BATCH_SIZE"]):
                            sample_mask_indices_val = []  # see comment above for sample_mask_indices
                            # no return value from apply_masks, everything is updated by reference in the lists
                            apply_masks_timedir(X_batch_val[x], y_batch_val[x], ref_samples, hp_dict, mask_variation,
                                                ytrue_multi_batch_val, batch_mask_indices_val, sample_mask_indices_val,
                                                mask_task=hp_dict["MSK_task"], log=log, heldout=True, task=hp_dict["task"])

                        # just used as metadata/bookkeeping/debugging, no analogue for training split
                        epoch_val_masks.append(batch_mask_indices_val)

                        # get ground truth one-hot distributions
                        for y in range(0, hp_dict["BATCH_SIZE"]):
                            if (y_batch_val[y][dir_idx] == 1):  # if it WAS reversed
                                ytrue_dist_bin_val = [0, 1]  # true
                            else:  # it was NOT reversed
                                ytrue_dist_bin_val = [1, 0]  # false
                            ytrue_bin_batch_val.append(
                                ytrue_dist_bin_val)

                        # convert label lists to pytorch tensors
                        ytrue_bin_batch_val = np.array(ytrue_bin_batch_val)
                        ytrue_multi_batch_val = np.array(ytrue_multi_batch_val)
                        ytrue_bin_batch_val = torch.from_numpy(ytrue_bin_batch_val).float()
                        ytrue_multi_batch_val = torch.from_numpy(ytrue_multi_batch_val).float()

                        # returns predictions for CLS then MSK task
                        ypred_bin_batch_val, ypred_multi_batch_val = model(X_batch_val, batch_mask_indices_val)

                        # get accuracy/precision stats for validation samples
                        for batch_idx, bin_pred in enumerate(
                                ypred_bin_batch_val):  # for each 2 dimensional output vector for the CLS task
                            bin_true = ytrue_bin_batch_val[batch_idx]
                            true_idx = torch.argmax(bin_true)
                            pred_idx = torch.argmax(bin_pred)
                            if (true_idx == pred_idx):
                                bin_correct_val[true_idx] += 1  # either 0 or 1 was the correct choice, so count it
                            if (true_idx == 0):
                                bin_neg_count_val += 1

                        # force to floats to avoid pytorch errors
                        ypred_bin_batch_val = ypred_bin_batch_val.float()
                        ypred_multi_batch_val = ypred_multi_batch_val.float()

                        # losses for CLS and MSK, obviously not backprop'd, just for benchmarking/analysis
                        loss_bin_val = criterion_bin(ypred_bin_batch_val, ytrue_bin_batch_val)
                        loss_multi_val = criterion_multi(ypred_multi_batch_val, ytrue_multi_batch_val)

                        # total loss depends on whether we're doing simultaneous training
                        if hp_dict["task"] == "CLS_only":
                            loss = loss_bin_val  # toy example for just same-genre task
                            acc = get_accuracy(ypred_bin_batch_val, ytrue_bin_batch_val, hp_dict["CLS_task"], log)
                        elif hp_dict["task"] == "MSK_only":
                            loss = loss_multi_val
                            acc = get_accuracy(ypred_multi_batch_val, ytrue_multi_batch_val, hp_dict["MSK_task"], log)

                        # see training split above for comments on all these variables
                        elif hp_dict["task"] == "both":
                            loss = (loss_bin_val * CLS_task_weight) + (
                                    loss_multi_val * MSK_task_weight)
                            epoch_vbin_loss += loss_bin_val.item()
                            epoch_vmulti_loss += loss_multi_val.item()
                            acc1 = get_accuracy(ypred_bin_batch_val, ytrue_bin_batch_val, hp_dict["CLS_task"], log)
                            acc2 = get_accuracy(ypred_multi_batch_val, ytrue_multi_batch_val, hp_dict["MSK_task"], log)

                        val_loss += loss.item()
                        if (hp_dict["task"] == "both"):
                            val_acc += acc1.item()
                            val_acc2 += acc2.item()
                        else:
                            val_acc += acc.item()
                            val_acc2 += 0

            print(
                f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f} | Acc2: {epoch_acc2 / len(train_loader):.3f}')
            print("Epoch bin training stats:\n")
            print("correct counts for this epoch: " + str(bin_correct_train))
            print("bin neg sample count: " + str(bin_neg_count_train))
            print("number of samples: " + str(len(train_loader)))

            log.write(
                f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f} | Acc2: {epoch_acc2 / len(train_loader):.3f}')
            if (hp_dict["task"] == "both"):
                print("Bin loss was " + str(epoch_tbin_loss / len(train_loader)) + " and multi loss was " + str(
                    epoch_tmulti_loss / len(train_loader)))
                log.write("Bin loss was " + str(epoch_tbin_loss / len(train_loader)) + " and multi loss was " + str(
                    epoch_tmulti_loss / len(train_loader)))
            if val_X is not None:
                print(
                    f'Validation: | Loss: {val_loss / len(val_loader):.5f} | Acc: {val_acc / len(val_loader):.3f} | Acc2: {val_acc2 / len(val_loader):.3f}')
                log.write(
                    f'Validation: | Loss: {val_loss / len(val_loader):.5f} | Acc: {val_acc / len(val_loader):.3f} | Acc2: {val_acc2 / len(val_loader):.3f}')

                if (hp_dict["task"] == "both"):
                    print(
                        "Bin val loss was " + str(epoch_vbin_loss / len(val_loader)) + " and multi val loss was " + str(
                            epoch_vmulti_loss / len(val_loader)))
                    log.write("Bin loss was " + str(epoch_vbin_loss / len(val_loader)) + " and multi loss was " + str(
                        epoch_vmulti_loss / len(val_loader)))
                print("Epoch bin val stats:\n")
                print("correct counts for this epoch: " + str(bin_correct_val))
                print("bin neg sample count: " + str(bin_neg_count_val))
                print("number of samples: " + str(len(val_loader)))
                # print("epoch val masks:" + str(epoch_val_masks) + "\n\n")

        if (save_model):
            modelcount = 0
            model_path = PRETRAIN_TIMEDIR_MODEL_PATH + str(hp_dict["dataset"]) + "/" + str(hp_dict["task"]) + "/states_" + str(training_fold) + str(
                modelcount) + ".pt"
            while (os.path.exists(model_path)):
                modelcount += 1
                model_path = PRETRAIN_TIMEDIR_MODEL_PATH + str(hp_dict["dataset"]) + "/" + str(hp_dict["task"]) + "/states_" + str(training_fold) + str(
                    modelcount) + ".pt"

            torch.save(model.state_dict(), model_path)
            model_path = PRETRAIN_TIMEDIR_MODEL_PATH + str(hp_dict["dataset"]) + "/" + str(hp_dict["task"]) + "/full_" + str(training_fold) + str(
                modelcount) + ".pt"
            torch.save(model, model_path)

    log.close()
