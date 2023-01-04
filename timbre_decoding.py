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

LR_def = 0.0001  # defaults, should normally be set by command line
printed_count = 0
val_printed_count = 0
# torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        thiscount = None  # gets changed if a count is passed as a command line argument
        # get command line arguments and options
        opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
        args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
        print("Command line opts: " + str(opts))
        print("Command line args: " + str(args))

        # text description of this job
        if "-m" in opts:
            # -m "this is the description of the run" will be at the end of the command line call
            idx = opts.index("-m")
            run_desc = args[idx]
        else:
            run_desc = None

        # what task was the pretrained model trained on? either "both", "CLS_only," "MSK_only," or "fresh"
        if "-pretrain_task" in opts:
            idx = opts.index("-pretrain_task")
            pretrain_task = args[idx]
        else:
            print(
                "provide the task the desired pretrained weights were trained on, i.e \"-pretrain_task both\" or \"-pretrain_idx fresh\" if training a fresh model. Choices are both, CLS_only, MSK_only, fresh")
            quit(0)

        # an index is assigned to each model during pretraining, and saved with that index in its name. that is how we now find and load a saved model.
        if "-pretrain_idx" in opts:
            idx = opts.index("-pretrain_idx")
            # this value will be "fresh" rather than a number if we want to train a fresh model
            # can probably keep this as a string since its just for file paths
            pretrain_idx = args[idx]
        else:
            print(
                "assign a pretrain model index to load, i.e \"-pretrain_idx 5\" or \"-pretrain_idx fresh\" if training a fresh model. choices are 0 to 11 inclusive")
            quit(0)

        # do we want to freeze pretrained weights or update them with the new output layer
        freeze_pretrained = False
        if "-freeze_pretrained" in opts:
            idx = opts.index("-freeze_pretrained")
            if args[idx] == "True":
                freeze_pretrained = True

        # train on gpu or not, not implemented yet
        if "-gpu" in opts:
            idx = opts.index("-gpu")
            gpunum = args[idx]  # currently only works if only one gpu is given
            device = torch.device("cuda:" + str(gpunum))
        else:
            device = "cpu"

        # index in job submission script, indicates the heldout run
        if "-heldout_run" in opts:
            # count and thiscount can be read as the index of the heldout run
            idx = opts.index("-heldout_run")
            thiscount = int(args[idx])
        else:
            thiscount = 0
        held_start = (600 + (400 * thiscount))
        held_range = range(held_start, held_start + 400)

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

        today = datetime.date.today()
        now = datetime.datetime.now()
        ##############################  SET PARAMETERS  ##############################
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dictionary of hyperparameters, eventually should probably come from command line
        hp_dict = {

            "pretrain_task": pretrain_task,  # passed in by command line
            "CLS_task": "Timbre_only",  # same_genre or nextseq
            "num_CLS_labels": 2, # either 2 or 4, only 4 if we're decoding HC/HT/IC/IT
            "MSK_task": None, # no MSK task in finetuning for now
            "COOL_DIVIDEND": COOL_DIVIDEND,
            "NUM_TOKENS": NUM_TOKENS,
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
            "data_dir": "2023-01-02-5TR",  # yyyy-mm-dd-nTR
            "include imagined":True, # needs to be set per dataset, manually, for now
            # Manually set the hemisphere and iteration number of the dataset you want to use in that folder
            "hemisphere": "left",
            "count": str(thiscount),  # count and thiscount can be read as the index of the heldout run
            "max_sample_length": 5,  # manually set max_seq_length used in data creation, does not include CLS token

            "within_subject": 1,
            # this doesn't really do anything now that inputs aren't paired, but it does serve as a reminder that the training data was created within subject (or not, potentially in the future)
            "num_subjects": 17,  # nakai et al's music genre dataset had 5 subjects
            "heldout_run": thiscount,
            "held_start": held_start,  # calculated above when handling command line args
            "held_range": held_range
        }

        # paths for loading pretrained model/weights
        # env is defined in Constants.py
        if env == "local":
            pretrained_model_states = "/Volumes/External/opengenre/preproc/trained_models/timedir/" + pretrain_task + "/states_" + pretrain_idx + ".pt"
            data_path = "/Volumes/External/pitchclass/finetuning/sametimbre/datasets/2/"
        if env=="discovery":
            pretrained_model_states = "/isi/music/auditoryimagery2/seanthesis/timedir/pretrain/trained_models/" + hp_dict["pretrain_task"] + "/states_" + pretrain_idx + "0.pt"
            data_path = "/isi/music/auditoryimagery2/seanthesis/timedir/finetune/datasets/"+hp_dict["data_dir"]+"/"

        # set up logfile, FINETUNE_TIMBRE_LOG_PATH is defined in Constants.py
        today_dir = FINETUNE_TIMBRE_LOG_PATH + str(hp_dict["CLS_task"]) + "/" + str(today) + "/"
        if not (os.path.exists(today_dir)):
            os.mkdir(today_dir)
        if (thiscount != None):
            # include the index of the heldout run, if there is one, to differentiate logs
            logcount = thiscount
        else:
            logcount = 0
        logfile = today_dir + "finetunelog_" + str(logcount) + ".txt"

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
        with open(data_path + "all_X.p", "rb") as samples_fp:
            train_X = pickle.load(samples_fp)
        with open(data_path + "all_y.p", "rb") as labels_fp:
            train_y_detailed = pickle.load(labels_fp)
        with open(data_path + "all_val_X.p", "rb") as valsamples_fp:
            val_X = pickle.load(valsamples_fp)
        with open(data_path + "all_val_y.p", "rb") as vallabels_fp:
            val_y_detailed = pickle.load(vallabels_fp)

        # the loaded labels are detailed tuples
        # detailed_label = (timbre, cond, subid, timbre, run_n, cycle_n)
        # put timbre in there twice on accident, whatever
        # timbre is either "Clarinet" or "Trumpet
        # cond is either "Heard" or "Imagined"
        train_y = []
        val_y = []

        for detailed_label in train_y_detailed:
            timbre = detailed_label[0]  # True/False is the first thing in the tuple
            condition = detailed_label[1]
            this_label = make_timbre_decode_label(timbre, condition, hp_dict["CLS_task"])

            train_y.append(this_label)

        for detailed_val_label in val_y_detailed:
            timbre = detailed_val_label[0]
            condition = detailed_val_label[1]
            this_label = make_timbre_decode_label(timbre, condition, hp_dict["CLS_task"])

            val_y.append(this_label)

        # train_X has shape (timesteps, max_length, voxel_dim)
        num_samples = len(train_X)
        max_length = len(train_X[0]) #should be seq_len + 1
        print("max length is "+str(max_length)+" and hpdict is "+str(hp_dict["max_sample_length"]))
        # assert (max_length == (hp_dict["max_sample_length"] + 1))
        voxel_dim = len(train_X[0][0])
        print("voxel dim is "+str(voxel_dim))
        print("num samples is "+str(num_samples))
        src_pad_sequence = [0] * voxel_dim # this is legacy crap that probably needs to be removed, but is currently needed

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
        train_loader = DataLoader(dataset=finetune_data, batch_size=hp_dict["BATCH_SIZE"],
                                  shuffle=True)  # make the DataLoader object
        val_loader = DataLoader(dataset=val_data, batch_size=hp_dict["BATCH_SIZE"], shuffle=False)

        # create the model
        model = Transformer(num_CLS_labels=hp_dict["num_CLS_labels"], num_genres=10, src_pad_sequence=src_pad_sequence, max_length=12,
                            voxel_dim=voxel_dim, ref_samples=None, mask_task=None, print_flag=0)

        # if we want to load pretrained weights:
        if pretrain_task != "fresh":
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

        criterion_CLS = nn.CrossEntropyLoss()
        # training_tensors is either everything or only the new layers for finetuning when freeze_pretrained is True
        optimizer = optim.Adam(training_tensors, lr=LR_def, betas=(0.9, 0.999), weight_decay=0.0001)

        ####################################################################################
        # FINALLY START TRAINING
        for e in range(0, hp_dict["EPOCHS"]):
            # 0'th index is the number of times the model was correct when the ground truth was 0, and when ground truth was 1
            CLS_correct_train = [0, 0]
            CLS_correct_val = [0, 0]
            CLS_neg_count_train = 0  # count the number of training samples where 0 was the correct answer
            CLS_neg_count_val = 0  # count the number of validation samples where 0 was the correct answer

            random.seed(seed + e)
            torch.manual_seed(seed + e)
            np.random.seed(seed + e)
            model.train()  # sets model status, doesn't actually start training
            # need the above every epoch because the validation part below sets model.eval()
            epoch_loss = 0
            epoch_acc = 0

            # LOOP THROUGH BATCHES WITHIN EPOCH
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.float()
                y_batch = y_batch.float()
                # X_batch, y_batch = X_batch.to(hp_dict["device"]), y_batch.to(hp_dict["device"])
                ytrue_CLS_batch = []  # list of batch targets for binary classification task

                optimizer.zero_grad()  # reset gradient to zero before each mini-batch
                for j in range(0, hp_dict["BATCH_SIZE"]):
                    ytrue_CLS_batch.append(y_batch[j].tolist()) # i dont really need to do this, but it's legacy/holdover crap

                # convert label lists to pytorch tensors
                ytrue_CLS_batch = np.array(ytrue_CLS_batch)
                ytrue_CLS_batch = torch.from_numpy(ytrue_CLS_batch).float()

                # passing finetune as the second parameter will skip all the mask token related stuff from pretraining
                # mask_indices set to finetune sends the CLS token to a different output layer than pretraining would
                ypred_CLS_batch = model(X_batch, mask_indices="finetune")
                ypred_CLS_batch = ypred_CLS_batch.float()
                loss = criterion_CLS(ypred_CLS_batch, ytrue_CLS_batch)
                acc = get_accuracy(ypred_CLS_batch, ytrue_CLS_batch, hp_dict["CLS_task"], log)
                epoch_loss += loss.item()
                epoch_acc += acc.item()

                # get some stats on this batch
                for batch_idx, CLS_pred in enumerate(
                        ypred_CLS_batch):  # for each 2 dimensional output vector for the binary task
                    CLS_true = ytrue_CLS_batch[batch_idx]
                    true_idx = torch.argmax(CLS_true)
                    pred_idx = torch.argmax(CLS_pred)
                    if (true_idx == pred_idx):
                        CLS_correct_train[true_idx] += 1  # either 0 or 1 was the correct choice, so count it
                    if (true_idx == 0):
                        CLS_neg_count_train += 1

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
                    ytrue_CLS_batch_val = []  # list of batch targets for binary classification task

                    for j in range(0, hp_dict["BATCH_SIZE"]):
                        ytrue_CLS_batch_val.append(y_batch_val[j].tolist())  # legacy crap doesn't really need to happen

                    # convert label lists to pytorch tensors
                    ytrue_CLS_batch_val = np.array(ytrue_CLS_batch_val)
                    ytrue_CLS_batch_val = torch.from_numpy(ytrue_CLS_batch_val).float()

                    # passing finetune as the second parameter will skip all the mask token related stuff from pretraining
                    ypred_CLS_batch_val= model(X_batch_val, mask_indices="finetune")

                    # get accuracy stats for validation samples
                    for batch_idx, CLS_pred in enumerate(
                            ypred_CLS_batch_val):  # for each 2 dimensional output vector for the binary task
                        CLS_true = ytrue_CLS_batch_val[batch_idx]
                        true_idx = torch.argmax(CLS_true)
                        pred_idx = torch.argmax(CLS_pred)
                        if (true_idx == pred_idx):
                            CLS_correct_val[true_idx] += 1  # either 0 or 1 was the correct choice, so count it
                        if (true_idx == 0):
                            CLS_neg_count_val += 1

                    ypred_CLS_batch_val = ypred_CLS_batch_val.float()
                    loss = criterion_CLS(ypred_CLS_batch_val, ytrue_CLS_batch_val)
                    acc = get_accuracy(ypred_CLS_batch_val, ytrue_CLS_batch_val, hp_dict["CLS_task"], log)
                    val_loss += loss.item()
                    val_acc += acc.item()

            # end epoch
            # print training stats for this epoch
            print("Epoch bin training stats:")
            print(
                f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
            print("correct counts for this epoch: " + str(CLS_correct_train))
            print("bin neg sample count: " + str(CLS_neg_count_train))
            print("number of samples: " + str(len(train_loader)))
            print("\n")
            print("Epoch bin val stats:")
            print(f'Epoch {e + 0:03}: | Loss: {val_loss / len(val_loader):.5f} | Acc: {val_acc / len(val_loader):.3f}')
            print("correct counts for this epoch's val split: " + str(CLS_correct_val))
            print("bin neg sample count: " + str(CLS_neg_count_val))
            print("number of samples: " + str(len(val_loader)))
        # end training

