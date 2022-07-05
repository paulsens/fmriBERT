import torch
import torch.nn as nn
import random
from random import randint
import numpy as np
from helpers import *
from voxel_transformer import *
from pitchclass_data import *
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from Constants import *
import sys
import os
import datetime

val_flag=1
random.seed(3)
mask_variation=False

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
        if "-count" in opts:
            thiscount=int(args[-2])
            held_start=(600 + (400 * thiscount))
            held_range=range(held_start, held_start+400)
        else:
            thiscount=None
            held_start=None
            held_range=None
        today = datetime.date.today()
        now = datetime.datetime.now()
        ##############################  SET PARAMETERS  ##############################
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #dictionary of hyperparameters, eventually should probably come from command line
        hp_dict={

            "task":"binaryonly",
            "binary":"same_genre",
            "mask_task":"reconstruction",
            "COOL_DIVIDEND" : COOL_DIVIDEND,
            "ATTENTION_HEADS" : ATTENTION_HEADS,
            "device" : str(device),
            "MSK_flag" : 1,
            "CLS_flag" : 1,
            "BATCH_SIZE" : 1,
            "EPOCHS" : EPOCHS,
            "LEARNING_RATE" : 0.0001,
            #Have to manually set the name of the folder whose training data you want to use, since there will be many
            "data_dir" : "2022-05-02",
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
        hp_dict["data_path"] = opengenre_preproc_path + "training_data/" + hp_dict["data_dir"] + "/"
        torch.set_default_dtype(torch.float64)

        #set up logfile, PRETRAIN_LOG_PATH is defined in Constants.py
        today_dir = PRETRAIN_LOG_PATH+str(today)+"/"
        # if not (os.path.exists(today_dir)):
        #     os.mkdir(today_dir)
        if(thiscount!=None):
            logcount=thiscount
        else:
            logcount=0
        logfile = today_dir + "pretrainlog_"+str(logcount)+".txt"
        while (os.path.exists(logfile)):
            logcount+=1
            logfile = today_dir + "pretrainlog_" + str(logcount) + ".txt"
        log = open(logfile,"w")
        log.write(str(now)+"\n")
        # run_desc is potentially given in command line call
        if(run_desc is not None):
            log.write(run_desc+"\n\n")
            print(run_desc+"\n\n")
        #write hyperparameters to log
        for hp in hp_dict.keys():
            log.write(str(hp)+" : "+str(hp_dict[hp])+"\n")

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
            val_loader = DataLoader(dataset=val_data, batch_size=hp_dict["BATCH_SIZE"], shuffle=True)
        log.write("voxel dim is "+str(voxel_dim)+"\n\n")
        MSK_token = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
        MSK_token = np.array(MSK_token)
        MSK_token = torch.from_numpy(MSK_token)

        same_genre_labels = 2 #two possible labels for same genre task, yes or no
        num_genres = 10 #from the training set

        src_pad_sequence = [0]*voxel_dim

        model = Transformer(next_sequence_labels=same_genre_labels, num_genres=num_genres, src_pad_sequence=src_pad_sequence, max_length=max_length, voxel_dim=voxel_dim, ref_samples=ref_samples, mask_task=hp_dict["mask_task"]).to(hp_dict["device"])
        model.to(hp_dict["device"])

        criterion_bin = nn.CrossEntropyLoss()
        if hp_dict["mask_task"]=="genre_decode":
            criterion_multi = nn.CrossEntropyLoss()
            get_multi_acc = True
        elif hp_dict["mask_task"]=="reconstruction":
            criterion_multi = nn.CrossEntropyLoss
            get_multi_acc = False
        optimizer = optim.Adam(model.parameters(), lr=hp_dict["LEARNING_RATE"], betas=(0.5,0.9), weight_decay=0.0001)

        for e in range(1, hp_dict["EPOCHS"]+1):
            model.train()  # sets model status, doesn't actually start training
                #need the above every epoch because the validation part below sets model.eval()
            epoch_loss = 0
            epoch_acc = 0

            for X_batch, y_batch in train_loader:
                batch_mask_indices = []
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
                    apply_masks(X_batch[x], y_batch[x], ref_samples, hp_dict, mask_variation, ytrue_multi_batch, sample_dists, ytrue_dist_multi1, ytrue_dist_multi2, batch_mask_indices, sample_mask_indices, log, heldout=False)
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
                log.write("ypred_multi_batch has shape "+str(ypred_multi_batch.shape)+"\n and ytrue_multi_batch has shape "+str(ytrue_multi_batch.shape))
                log.write("For binary classification, predictions are "+str(ypred_bin_batch)+" and true labels are "+str(ytrue_bin_batch)+"\n")
                loss_bin = criterion_bin(ypred_bin_batch, ytrue_bin_batch)
                log.write("The loss in that case was "+str(loss_bin)+"\n")
                #log.write("For genre classification, predictions are "+str(ypred_multi_batch)+" and true labels are "+str(ytrue_multi_batch)+"\n")
                loss_multi = criterion_multi(ypred_multi_batch, ytrue_multi_batch)
                #log.write("The loss in that case was "+str(loss_multi)+"\n\n")

                if hp_dict["task"] == "binaryonly":
                    loss = loss_bin #toy example for just same-genre task
                    acc = get_accuracy(ypred_bin_batch, ytrue_bin_batch, log)
                elif hp_dict["task"] == "multionly":
                    loss = loss_multi
                    if(get_multi_acc):
                        acc = get_accuracy(ypred_multi_batch, ytrue_multi_batch, log)
                    else:
                        acc = 0 #placeholder until i figure out how to do accuracy for non-genre-decoding tasks
                else:
                    loss = (loss_bin+loss_multi)/2 #as per devlin et al, loss is the average of the two tasks' losses
                    acc = 0
                #log.write("The total loss this iteration was "+str(loss)+"\n\n")

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                #log.write("added "+str(acc.item())+" to epoch_acc")

            # now calculate validation loss/acc, turn off gradient
            if val_X is not None:
                with torch.no_grad():
                    val_loss=0
                    val_acc=0
                    for X_batch_val, y_batch_val in val_loader:
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
                            apply_masks(X_batch_val[x], y_batch_val[x], ref_samples, hp_dict, mask_variation,   ytrue_multi_batch_val, sample_dists_val, ytrue_dist_multi1_val, ytrue_dist_multi2_val, batch_mask_indices_val, sample_mask_indices_val, log, heldout=False)
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

                        # returns predictions for binary class and multiclass, in that order
                        ypred_bin_batch_val, ypred_multi_batch_val = model(X_batch_val, batch_mask_indices_val)
                        log.write("ypred_multi_batch_val has shape " + str(
                            ypred_multi_batch.shape) + "\n and ytrue_multi_batch_val has shape " + str(
                            ytrue_multi_batch.shape))
                        # log.write("For binary classification, predictions are "+str(ypred_bin_batch)+" and true labels are "+str(ytrue_bin_batch)+"\n")
                        loss_bin_val = criterion_bin(ypred_bin_batch_val, ytrue_bin_batch_val)
                        # log.write("The loss in that case was "+str(loss_bin)+"\n")
                        # log.write("For genre classification, predictions are "+str(ypred_multi_batch)+" and true labels are "+str(ytrue_multi_batch)+"\n")
                        loss_multi = criterion_multi(ypred_multi_batch_val, ytrue_multi_batch_val)
                        # log.write("The loss in that case was "+str(loss_multi)+"\n\n")

                        if hp_dict["task"] == "binaryonly":
                            loss = loss_bin  # toy example for just same-genre task
                            acc = get_accuracy(ypred_bin_batch, ytrue_bin_batch, log)
                        elif hp_dict["task"] == "multionly":
                            loss = loss_multi
                            if(get_multi_acc):
                                acc = get_accuracy(ypred_multi_batch, ytrue_multi_batch, log)
                            else: acc = 0
                        else:
                            loss = (
                                               loss_bin + loss_multi) / 2  # as per devlin et al, loss is the average of the two tasks' losses
                            acc = 0
                        # log.write("The total loss this iteration was "+str(loss)+"\n\n")

                        val_loss += loss.item()
                        val_acc += acc.item()


            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

            log.write(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
            if val_X is not None:
                print(f'Validation: | Loss: {val_loss/len(val_loader):.5f} | Acc: {val_acc/len(val_loader):.3f}')
                log.write(f'Validation: | Loss: {val_loss/len(val_loader):.5f} | Acc: {val_acc/len(val_loader):.3f}')

            #print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')
            #log.write(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')


        # modelcount=0
        # model_path = hp_dict["data_path"]+str(hp_dict["task"])+"_states_"+str(thiscount)+str(modelcount)+".pt"
        # while(os.path.exists(model_path)):
        #     modelcount+=1
        #     model_path = hp_dict["data_path"] + str(hp_dict["task"]) + "_states_" + str(thiscount) + str(
        #         modelcount) + ".pt"
        #
        # torch.save(model.state_dict(),model_path)
        # model_path = hp_dict["data_path"]+str(hp_dict["task"])+"_full_"+str(thiscount)+str(modelcount)+".pt"
        # torch.save(model,model_path)

    log.close()
