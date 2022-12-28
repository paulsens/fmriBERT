# mostly a copy of pretrain.py with adjustments for pure evaluation
# index of saved from from 0 to 11 inclusive
index = 0
# task is either binaryonly, multionly, or both
evaltask = "multionly"
#basepath = "/Volumes/External/opengenre/saveloadtest/2/"
#state_path = "/Volumes/External/opengenre/preproc/trained_models/"+str(evaltask)+"/sep26/"
#state_file = state_path+"states_"+str(index)+"0.pt"
#model_file = "/isi/music/auditoryimagery2/seanthesis/opengenre/preproc/trained_models/binaryonly/oct1/full_52.pt"
model_file =  "/Volumes/External/opengenre/round1/final/"+evaltask+"/states_"+str(index)+"0.pt"

import torch
import torch.nn as nn
import random
from random import randint
import numpy as np
from helpers import *
from voxel_transformer_print import *
from pitchclass_data import *
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from Constants import *
import sys
import os
import datetime
#torch.use_deterministic_algorithms(True)

debug=1
val_flag=1
seed=3
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
mask_variation=True
valid_accuracy=True
bintask_weight = 0.5
multitask_weight = 1-bintask_weight
LR_def=0.0001 #defaults, should normally be set by command line
printed_count=0
val_printed_count=0

if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        thiscount=None #gets changed if a count is passed as a command line argument
        #get command line arguments and options
        opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
        args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
        if "-m" in opts:
            # -m "this is the description of the run" will be at the end of the command line call
            run_desc = args[-1]
        else:
            run_desc = None
        if "-gpu" in opts:
            gpunum=args[-3] #currently only works if only one gpu is given
            device = torch.device("cuda:"+str(gpunum))
        else:
            device="cpu"
        #if "-count" in opts:
        if True:
            thiscount=index
            held_start=(600 + (400 * thiscount))
            held_range=range(held_start, held_start+400)
        #else:
            # thiscount=None
            # held_start=None
            # held_range=None
        #if "-LR" in opts:
            #LR = args[-3]
            LR = 0.0001
            if LR=="default":
                LR = LR_def #default value if nothing is passed by command line
            LR = float(LR)

        #if "-binweight" in opts:
            bintask_weight= 0.1
            if bintask_weight=="default":
                bintask_weight=1
            else:
                bintask_weight=float(bintask_weight)
            multitask_weight = 1 - bintask_weight  # if we're just averaging then these two weights would both be 0.5

            #current format for command line args is binweight LR count message, counting backward because fuck
            #need a better system for this, probably should just stop letting these parameters be optional
        today = datetime.date.today()
        now = datetime.datetime.now()
        ##############################  SET PARAMETERS  ##############################
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #dictionary of hyperparameters, eventually should probably come from command line
        hp_dict={

            "task":"multionly",
            "binary":"nextseq", #same_genre or nextseq
            "mask_task":"reconstruction",
            "COOL_DIVIDEND" : COOL_DIVIDEND,
            "ATTENTION_HEADS" : ATTENTION_HEADS,
            "device" : str(device),
            "MSK_flag" : 1,
            "CLS_flag" : 1,
            "BATCH_SIZE" : 1,
            "EPOCHS" : EPOCHS,
            "LEARNING_RATE" : LR, #set at top of this file or by command line argument
            #Have to manually set the name of the folder whose training data you want to use, since there will be many
            "data_dir" : "2022-06-12",
            #Manually set the hemisphere and iteration number of the dataset you want to use in that folder
            "hemisphere": "left",
            "count" : str(thiscount),
            #manually set max_seq_length used in data creation, in the input CLS+seq+SEP+seq this is the max length of seq
            "max_sample_length":5,
            "mask_variation":mask_variation,
            "within_subject":1,
            "num_subjects":5,
            "heldout_run":thiscount,
            "held_start":held_start, #first 600 indices of refsamples are from testruns
            "held_range":held_range
        }

        if(debug):
            print("LR is "+str(hp_dict["LEARNING_RATE"]))
            print("bintaskweight is "+str(bintask_weight))
            print("multitaskweight is "+str(multitask_weight))

        hp_dict["data_path"] = opengenre_preproc_path + "training_data/" + hp_dict["data_dir"] + "/"
        torch.set_default_dtype(torch.float32)

        #set up logfile, PRETRAIN_LOG_PATH is defined in Constants.py
        today_dir = PRETRAIN_LOG_PATH+str(today)+"/"
        # if not (os.path.exists(today_dir)):
        #     os.mkdir(today_dir)
        if(thiscount!=None):
            logcount=thiscount
        else:
            logcount=0
        #logfile = today_dir + "pretrainlog_"+str(logcount)+".txt"
        #logfile = basepath + "pretrainlog_"+str(logcount)+".txt"
        logfile=None

        # while (os.path.exists(logfile)):
        #     logcount+=1
        #     #logfile = today_dir + "pretrainlog_" + str(logcount) + ".txt"
        #     logfile = basepath + "pretrainlog_" + str(logcount) + ".txt"

        # log = open(logfile,"w")
        # log.write(str(now)+"\n")
        # run_desc is potentially given in command line call
        # if(run_desc is not None):
        #     log.write(run_desc+"\n\n")
        #     print(run_desc+"\n\n")
        #write hyperparameters to log
        # for hp in hp_dict.keys():
        #     log.write(str(hp)+" : "+str(hp_dict[hp])+"\n")

        #load samples and labels
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_samples"+hp_dict["count"]+".p", "rb") as samples_fp:
            train_X = pickle.load(samples_fp)
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_labels"+hp_dict["count"]+".p", "rb") as labels_fp:
            train_Y = pickle.load(labels_fp)
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_refsamples.p", "rb") as refsamples_fp:
            ref_samples = pickle.load(refsamples_fp)
            #ref_samples has length 5400, 1080*5subjects.
                # 1080 = 20*6testruns + 80*12trainingruns
        #load valsamples and vallabels
        if(val_flag==0):
            val_X=None
            val_Y=None
        else:
            with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_valsamples"+hp_dict["count"]+".p", "rb") as samples_fp:
                val_X = pickle.load(samples_fp)
            with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_vallabels"+hp_dict["count"]+".p", "rb") as labels_fp:
                val_Y = pickle.load(labels_fp)


        #train_X has shape (timesteps, max_length, voxel_dim)
        num_samples = len(train_X)
        max_length = len(train_X[0]) #should be max_length*2 + 2
        assert (max_length == (hp_dict["max_sample_length"]*2 +2))

        voxel_dim = len(train_X[0][0])
        print("voxel dim is "+str(voxel_dim))
        print("num samples is "+str(num_samples))
        #convert to numpy arrays
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
        print("train x has shape "+str(train_X.shape))
        if val_X is not None:
            val_X = np.array(val_X)
            val_Y = np.array(val_Y)

        #convert to tensors
        train_X = torch.from_numpy(train_X)
        train_Y = torch.from_numpy(train_Y)
        if val_X is not None:
            val_X = torch.from_numpy(val_X)
            val_Y = torch.from_numpy(val_Y)

        all_data = TrainData(train_X, train_Y) #make the TrainData object
        if val_X is not None:
            val_data = TrainData(val_X, val_Y) #make TrainData object for validation data

        #train_val_dataset defined in helpers, val_split defined in Constants
        #datasets = train_val_dataset(all_data, val_split)

        train_loader = DataLoader(dataset=all_data, batch_size=hp_dict["BATCH_SIZE"], shuffle=True) #make the DataLoader object
        if val_X is not None:
            val_loader = DataLoader(dataset=val_data, batch_size=hp_dict["BATCH_SIZE"], shuffle=False)
       # log.write("voxel dim is "+str(voxel_dim)+"\n\n")
        MSK_token = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
        MSK_token = np.array(MSK_token)
        MSK_token = torch.from_numpy(MSK_token)

        binary_task_labels = 2 #two possible labels for same genre task, yes or no
        num_genres = 10 #from the training set
        log=None
        src_pad_sequence = [0]*voxel_dim

        model = Transformer(next_sequence_labels=binary_task_labels, num_genres=num_genres, src_pad_sequence=src_pad_sequence, max_length=max_length, voxel_dim=voxel_dim, ref_samples=ref_samples, mask_task=hp_dict["mask_task"]).to(hp_dict["device"])
        model.load_state_dict(torch.load(model_file))
        model = model.float()
        #model.to(hp_dict["device"])
        model.print_flag = 0
        print(model)
        quit(0)
        criterion_bin = nn.CrossEntropyLoss()
        if hp_dict["mask_task"]=="genre_decode":
            criterion_multi = nn.CrossEntropyLoss()
            get_multi_acc = True
        elif hp_dict["mask_task"]=="reconstruction":
            criterion_multi = nn.MSELoss()
            get_multi_acc = False
        optimizer = optim.Adam(model.parameters(), lr=hp_dict["LEARNING_RATE"], betas=(0.9,0.999), weight_decay=0.0001)

        for e in range(1, hp_dict["EPOCHS"]+1):
            epoch_val_masks=[]
            #0'th index is the number of times the model was correct when the ground truth was 0, and when ground truth was 1
            bin_correct_train = [0,0]
            bin_correct_val = [0,0]
            bin_neg_count_train = 0 #count the number of training samples where 0 was the correct answer
            bin_neg_count_val = 0 #count the number of validation samples where 0 was the correct answer

            random.seed(seed+e)
            torch.manual_seed(seed+e)
            np.random.seed(seed+e)
            model.eval()  # sets model status, doesn't actually start training
                #need the above every epoch because the validation part below sets model.eval()
            epoch_loss = 0
            epoch_acc = 0
            epoch_acc2 = 0

            for X_batch, y_batch in train_loader:
                print("length of train loader is "+str(len(train_loader)))
                batch_mask_indices = []
                X_batch=X_batch.float()
                y_batch=y_batch.float()
                X_batch, y_batch = X_batch.to(hp_dict["device"]), y_batch.to(hp_dict["device"])
                ytrue_bin_batch = [] #list of batch targets for binary classification task
                ytrue_multi_batch = [] #list of batch targets for multi-classification task

                optimizer.zero_grad() #reset gradient to zero before each mini-batch
                for x in range(0,hp_dict["BATCH_SIZE"]):
                    sample_mask_indices = []  # will either have 1 or 2 ints in it
                    sample_dists = [] #will be appended to ytrue_multi_batch
                    ytrue_dist_multi1 = np.zeros((10,))  # we want a one-hot probability distrubtion over the 10 genre labels
                    ytrue_dist_multi2 = np.zeros((10,))  # only used when this sample gets two masks/replacements
                    #no return value from apply_masks, everything is updated by reference in the lists
                    apply_masks(X_batch[x], y_batch[x], ref_samples, hp_dict, mask_variation, ytrue_multi_batch, sample_dists, ytrue_dist_multi1, ytrue_dist_multi2, batch_mask_indices, sample_mask_indices, mask_task=hp_dict["mask_task"], log=None, heldout=False)

                for y in range(0,hp_dict["BATCH_SIZE"]):
                    if(y_batch[y][0]):
                        ytrue_dist_bin = [0,1] #true, they are the same genre
                    else:
                        ytrue_dist_bin = [1,0] #false
                    ytrue_bin_batch.append(ytrue_dist_bin) #should give a list BATCHSIZE many same_genre boolean targets

                #convert label lists to pytorch tensors
                ytrue_bin_batch = np.array(ytrue_bin_batch)
                ytrue_multi_batch = np.array(ytrue_multi_batch)

                #batch_mask_indices = np.array(batch_mask_indices)

                ytrue_bin_batch = torch.from_numpy(ytrue_bin_batch).float()
                ytrue_multi_batch = torch.from_numpy(ytrue_multi_batch).float()

                #batch_mask_indices = torch.from_numpy(batch_mask_indices).float()

                #send this stuff to device
                ytrue_bin_batch.to(hp_dict["device"])
                #batch_mask_indices.to(hp_dict["device"])

                #returns predictions for binary class and multiclass, in that order
                ypred_bin_batch,ypred_multi_batch = model(X_batch, batch_mask_indices)
                for batch_idx, bin_pred in enumerate(ypred_bin_batch): #for each 2 dimensional output vector for the binary task
                    bin_true=ytrue_bin_batch[batch_idx]
                    true_idx=torch.argmax(bin_true)
                    pred_idx=torch.argmax(bin_pred)
                    if(true_idx==pred_idx):
                        bin_correct_train[true_idx]+=1 #either 0 or 1 was the correct choice, so count it
                    if(true_idx==0):
                        bin_neg_count_train+=1

                ypred_bin_batch = ypred_bin_batch.float()
                ypred_multi_batch = ypred_multi_batch.float()
                #log.write("ypred_multi_batch has shape "+str(ypred_multi_batch.shape)+"\n and ytrue_multi_batch has shape "+str(ytrue_multi_batch.shape))
                #log.write("For binary classification, predictions are "+str(ypred_bin_batch)+" and true labels are "+str(ytrue_bin_batch)+"\n")
                loss_bin = criterion_bin(ypred_bin_batch, ytrue_bin_batch)
                #log.write("The loss in that case was "+str(loss_bin)+"\n")
                #log.write("For genre classification, predictions are "+str(ypred_multi_batch)+" and true labels are "+str(ytrue_multi_batch)+"\n")
                loss_multi = criterion_multi(ypred_multi_batch, ytrue_multi_batch)
                #log.write("The loss in that case was "+str(loss_multi)+"\n\n")
                if printed_count < 1:
                    #print("sample "+str(printed_count)+": "+str(X_batch)+"\n\n")
                    #print("ypred "+str(printed_count)+": "+str(ypred_bin_batch)+"\n\n")
                    #print("ytrue "+str(printed_count)+": "+str(ytrue_bin_batch)+"\n\n")
                    printed_count+=1
                    model.print_flag=0

                if hp_dict["task"] == "binaryonly":
                    loss = loss_bin #toy example for just same-genre task
                    acc = get_accuracy(ypred_bin_batch, ytrue_bin_batch, hp_dict["binary"], log)
                elif hp_dict["task"] == "multionly":
                    loss = loss_multi
                    print_choice=random.randint(0,100)
                    #if(print_choice>98):
                        #print("batch loss is "+str(loss))
                        #print("ypred is "+str(ypred_multi_batch))
                        #print("ytrue is "+str(ytrue_multi_batch))
                    acc = get_accuracy(ypred_multi_batch, ytrue_multi_batch, hp_dict["mask_task"], log)

                    # if(get_multi_acc):
                    #     acc = get_accuracy(ypred_multi_batch, ytrue_multi_batch, hp_dict["mask_task"], log)
                    # else:
                    #     acc = 0 #placeholder until i figure out how to do accuracy for non-genre-decoding tasks
                elif hp_dict["task"]=="both":
                    loss = (loss_bin*bintask_weight) + (loss_multi*multitask_weight) #as per devlin et al, loss is the average of the two tasks' losses, in which case both weights would be 0.5
                    acc1 = get_accuracy(ypred_bin_batch, ytrue_bin_batch, hp_dict["binary"], log)
                    acc2 = get_accuracy(ypred_multi_batch, ytrue_multi_batch, hp_dict["mask_task"], log)
                #log.write("The total loss this iteration was "+str(loss)+"\n\n")

                #loss.backward()
                #optimizer.step()

                epoch_loss += loss.item()
                #the word valid here does not refer to validation, but rather is this task something we can obtain a valid accuracy for
                if(valid_accuracy):
                    if(hp_dict["task"]=="both"):
                        epoch_acc+=acc1.item()
                        epoch_acc2+=acc2.item()
                    else:
                        epoch_acc += acc.item()
                        epoch_acc2 += 0
                # use this break if you only want one sample from the training data
                break
                # if not valid_accuracy:
                #     print("Accuracy is invalid.")
                #     log.write("Accuracy is invalid.")
                #log.write("added "+str(acc.item())+" to epoch_acc")

            # now calculate validation loss/acc, turn off gradient
            # print("Model params before val split:\n")
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name,param.data)

            if val_X is not None:
                count = 0
                model.eval()
                model.print_flag=0

                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                with torch.no_grad():
                    val_loss=0
                    val_acc=0
                    val_acc2=0
                    for X_batch_val, y_batch_val in val_loader:
                        X_batch_val=X_batch_val.float()
                        y_batch_val=y_batch_val.float()
                        batch_mask_indices_val = []
                        ytrue_bin_batch_val = []  # list of batch targets for binary classification task
                        ytrue_multi_batch_val = []  # list of batch targets for multi-classification task
                        X_batch_val, y_batch_val = X_batch_val.to(hp_dict["device"]), y_batch_val.to(hp_dict["device"])

                        for x in range(0, hp_dict["BATCH_SIZE"]):
                            sample_mask_indices_val = []  # will either have 1 or 2 ints in it
                            sample_dists_val = []  # will be appended to ytrue_multi_batch
                            ytrue_dist_multi1_val = np.zeros(
                                (10,))  # we want a one-hot probability distrubtion over the 10 genre labels
                            ytrue_dist_multi2_val = np.zeros(
                                (10,))  # only used when this sample gets two masks/replacements
                            # no return value from apply_masks, everything is updated by reference in the lists
                            apply_masks(X_batch_val[x], y_batch_val[x], ref_samples, hp_dict, mask_variation,   ytrue_multi_batch_val, sample_dists_val, ytrue_dist_multi1_val, ytrue_dist_multi2_val, batch_mask_indices_val, sample_mask_indices_val, mask_task=hp_dict["mask_task"], log=log, heldout=True)
                        print("val x is " + str(X_batch_val[0]))
                        epoch_val_masks.append(batch_mask_indices_val)

                        for y in range(0, hp_dict["BATCH_SIZE"]):
                            if (y_batch_val[y][0]):
                                ytrue_dist_bin_val = [0, 1]  # true, they are the same genre
                            else:
                                ytrue_dist_bin_val = [1, 0]  # false
                            ytrue_bin_batch_val.append(
                                ytrue_dist_bin_val)  # should give a list BATCHSIZE many same_genre boolean targets

                        # convert label lists to pytorch tensors
                        ytrue_bin_batch_val = np.array(ytrue_bin_batch_val)
                        ytrue_multi_batch_val = np.array(ytrue_multi_batch_val)
                        ytrue_bin_batch_val = torch.from_numpy(ytrue_bin_batch_val).float()
                        ytrue_multi_batch_val = torch.from_numpy(ytrue_multi_batch_val).float()
                        epoch_val_masks.append(batch_mask_indices_val)
                        # returns predictions for binary class and multiclass, in that order
                        print("Xbatchval is "+str(X_batch_val))
                        ypred_bin_batch_val, ypred_multi_batch_val = model(X_batch_val, batch_mask_indices_val)
                        count+=1
                        print(ypred_bin_batch_val)
                        print(ypred_multi_batch_val)
                        print(batch_mask_indices_val)
                        if(count==2):
                            quit(0)

                        #get accuracy stats for validation samples
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
                        ypred_multi_batch_val = ypred_multi_batch_val.float()
                        # log.write("ypred_multi_batch_val has shape " + str(
                        #     ypred_multi_batch_val.shape) + "\n and ytrue_multi_batch_val has shape " + str(
                        #     ytrue_multi_batch_val.shape))
                        # log.write("For binary classification, predictions are "+str(ypred_bin_batch)+" and true labels are "+str(ytrue_bin_batch)+"\n")
                        loss_bin_val = criterion_bin(ypred_bin_batch_val, ytrue_bin_batch_val)
                        # log.write("The loss in that case was "+str(loss_bin)+"\n")
                        # log.write("For genre classification, predictions are "+str(ypred_multi_batch)+" and true labels are "+str(ytrue_multi_batch)+"\n")
                        loss_multi_val = criterion_multi(ypred_multi_batch_val, ytrue_multi_batch_val)
                        # log.write("The loss in that case was "+str(loss_multi)+"\n\n")

                        if hp_dict["task"] == "binaryonly":
                            loss = loss_bin_val  # toy example for just same-genre task
                            acc = get_accuracy(ypred_bin_batch_val, ytrue_bin_batch_val, hp_dict["binary"],log)
                        elif hp_dict["task"] == "multionly":
                            loss = loss_multi_val
                            acc = get_accuracy(ypred_multi_batch_val, ytrue_multi_batch_val, hp_dict["mask_task"], log)

                            # if(get_multi_acc):
                            #     acc = get_accuracy(ypred_multi_batch, ytrue_multi_batch, log)
                            # else: acc = 0
                        elif hp_dict["task"]=="both":
                            loss = (loss_bin * bintask_weight) + (
                                        loss_multi * multitask_weight)  # as per devlin et al, loss is the average of the two tasks' losses, in which case both weights would be 0.5
                            acc1 = get_accuracy(ypred_bin_batch_val, ytrue_bin_batch_val, hp_dict["binary"],log)
                            acc2 = get_accuracy(ypred_multi_batch_val, ytrue_multi_batch_val, hp_dict["mask_task"], log)
                        # log.write("The total loss this iteration was "+str(loss)+"\n\n")

                        val_loss += loss.item()
                        if(valid_accuracy):
                            if(hp_dict["task"]=="both"):
                                val_acc += acc1.item()
                                val_acc2 += acc2.item()
                            else:
                                val_acc += acc.item()
                                val_acc2 += 0
                        if val_printed_count < 1:
                            print("val sample " + str(val_printed_count) + ": " + str(X_batch_val) + "\n\n")
                            print("val ypred " + str(val_printed_count) + ": " + str(ypred_bin_batch_val) + "\n\n")
                            print("val ytrue " + str(val_printed_count) + ": " + str(ytrue_bin_batch_val) + "\n\n")
                            val_printed_count += 1
                            model.print_flag = 0

            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f} | Acc2: {epoch_acc2/len(train_loader):.3f}')
            print("Epoch bin training stats:\n")
            print("correct counts for this epoch: "+str(bin_correct_train))
            print("bin neg sample count: "+str(bin_neg_count_train))
            print("number of samples: "+str(len(train_loader)))

           # log.write(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f} | Acc2: {epoch_acc2/len(train_loader):.3f}')
            if(hp_dict["task"]=="both"):
                print("Bin loss was "+str(loss_bin)+ " and multi loss was "+str(loss_multi))
               # log.write("Bin loss was "+str(loss_bin)+ " and multi loss was "+str(loss_multi))
            if val_X is not None:
                print(f'Validation: | Loss: {val_loss/len(val_loader):.5f} | Acc: {val_acc/len(val_loader):.3f} | Acc2: {val_acc2/len(val_loader):.3f}')
                #log.write(f'Validation: | Loss: {val_loss/len(val_loader):.5f} | Acc: {val_acc/len(val_loader):.3f} | Acc2: {val_acc2/len(val_loader):.3f}')

                if(hp_dict["task"]=="both"):
                    print("Bin val loss was " + str(loss_bin_val) + " and multi val loss was " + str(loss_multi_val))
                    #log.write("Bin loss was " + str(loss_bin_val) + " and multi loss was " + str(loss_multi_val))
                print("Epoch bin val stats:\n")
                print("correct counts for this epoch: " + str(bin_correct_val))
                print("bin neg sample count: " + str(bin_neg_count_val))
                print("number of samples: " + str(len(val_loader)))
                print("epoch val masks:" + str(epoch_val_masks) + "\n\n")

                # if not valid_accuracy:
                #     print("Accuracy is invalid.")
                #     log.write("Accuracy is invalid.")

            #print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')
            #log.write(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')

        # print("before saving model, model has params:\n")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        # modelcount=0
        # model_path = opengenre_preproc_path+"trained_models/"+str(hp_dict["task"])+"/oct1/"+"states_"+str(thiscount)+str(modelcount)+".pt"
        # while(os.path.exists(model_path)):
        #     modelcount+=1
        #     model_path = opengenre_preproc_path+"trained_models/"+str(hp_dict["task"])+"/oct1/"+"states_"+str(thiscount)+str(modelcount)+".pt"
        #
        # torch.save(model.state_dict(),model_path)
        # model_path = opengenre_preproc_path + "trained_models/" + str(hp_dict["task"]) + "/oct1/" + "full_" + str(thiscount) + str(
        #     modelcount) + ".pt"
        # torch.save(model,model_path)

    log.close()

