import torch
import torch.nn as nn
from random import randint
import numpy as np
from helpers import *
from voxel_transformer import *
from pitchclass_data import *
from torch.utils.data import DataLoader
import torch.optim as optim
from Constants import *
import sys
import os
import datetime

if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        #get command line arguments and options
        opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
        args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
        if "-m" in opts:
            # -m "this is the description of the run" will be at the end of the command line call
            run_desc = args[-1]
        else:
            run_desc = None
        today = datetime.date.today()
        now = datetime.datetime.now()
        ##############################  SET PARAMETERS  ##############################
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #dictionary of hyperparameters, eventually should probably come from command line
        hp_dict={
            "COOL_DIVIDEND" : COOL_DIVIDEND,
            "ATTENTION_HEADS" : ATTENTION_HEADS,
            "device" : "cpu",
            "MSK_flag" : 1,
            "CLS_flag" : 1,
            "BATCH_SIZE" : 1,
            "EPOCHS" : EPOCHS,
            "LEARNING_RATE" : 0.00005,
            #Have to manually set the name of the folder whose training data you want to use, since there will be many
            "data_dir" : "2022-04-15-13",
            #manually set max_seq_length used in data creation, in the input CLS+seq+SEP+seq this is the max length of seq
            "max_sample_length":5
        }
        hp_dict["data_path"] = opengenre_preproc_path + "training_data/" + hp_dict["data_dir"] + "/"
        torch.set_default_dtype(torch.float64)

        #set up logfile, PRETRAIN_LOG_PATH is defined in Constants.py
        today_dir = PRETRAIN_LOG_PATH+str(today)+"/"
        if not (os.path.exists(today_dir)):
            os.mkdir(today_dir)

        logcount=0
        logfile = today_dir + "pretrain_log_"+str(logcount)+".txt"
        while (os.path.exists(logfile)):
            logcount+=1
            logfile = today_dir + "pretrain_log_" + str(logcount) + ".txt"
        log = open(logfile,"w")
        logfile.write(str(now)+"\n")
        if(run_desc is not None):
            logfile.write(run_desc+"\n\n")
        #write hyperparameters to log
        for hp in hp_dict.keys():
            log.write(str(hp)+" : "+str(hp_dict[hp])+"\n")

        #load samples and labels
        with open(hp_dict["data_path"] + "samples.p", "rb") as samples_fp:
            train_X = pickle.load(samples_fp)
        with open(hp_dict["data_path"] + "labels.p", "rb") as labels_fp:
            train_Y = pickle.load(labels_fp)
        #train_X has shape (timesteps, max_length, voxel_dim)
        num_samples = len(train_X)
        max_length = len(train_X[0]) #should be max_length*2 + 2
        assert (max_length == (hp_dict["max_sample_length"]*2 +2))

        voxel_dim = len(train_X[0][0])

        #convert to numpy arrays
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
        #convert to tensors
        train_X = torch.from_numpy(train_X)
        train_Y = torch.from_numpy(train_Y)

        train_data = TrainData(train_X, train_Y) #make the TrainData object
        train_loader = DataLoader(dataset=train_data, batch_size=hp_dict["BATCH_SIZE"], shuffle=True) #make the DataLoader object
        print("voxel dim is "+str(voxel_dim))
        log.write("voxel dim is "+str(voxel_dim)+"\n\n")
        MSK_token = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
        MSK_token = np.array(MSK_token)
        MSK_token = torch.from_numpy(MSK_token)

        same_genre_labels = 2 #two possible labels for same genre task, yes or no
        num_genres = 10 #from the training set

        src_pad_sequence = [0]*voxel_dim

        model = Transformer(next_sequence_labels=same_genre_labels, num_genres=num_genres, src_pad_sequence=src_pad_sequence, max_length=max_length, voxel_dim=voxel_dim).to(hp_dict["device"])
        model.to(hp_dict["device"])

        criterion_bin = nn.CrossEntropyLoss()
        criterion_multi = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=hp_dict["LEARNING_RATE"], betas=(0.5,0.9), weight_decay=0.0001)

        model.train() #sets model status, doesn't actually start training
        for e in range(1, hp_dict["EPOCHS"]+1):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in train_loader:
                batch_mask_indices = []
                X_batch, y_batch = X_batch.to(hp_dict["device"]), y_batch.to(hp_dict["device"])
                ytrue_bin_batch = [] #list of batch targets for binary classification task
                ytrue_multi_batch = [] #list of batch targets for multi-classification task
                optimizer.zero_grad() #reset gradient to zero before each mini-batch
                for x in range(0,hp_dict["BATCH_SIZE"]):
                    mask_choice = randint(1,10) #pick a token to mask
                    if (mask_choice >= 6):
                        mask_choice +=1 #dont want to mask the SEP token at index 6, so 6-10 becomes 7-11
                        #each element in the batch has 3 values, same_genre boolean, first half genre, second half genre
                        #so if we're masking an element of the second half, the genre decoding label should be that half's genre
                        ytrue_multi_idx = y_batch[x][2]
                    else:
                        ytrue_multi_idx = y_batch[x][1]
                    ytrue_dist_multi = np.zeros((10,)) #we want a one-hot probability distrubtion over the 10 genre labels
                    ytrue_dist_multi[ytrue_multi_idx]=1 #set all the probability mass on the true index
                    ytrue_multi_batch.append(ytrue_dist_multi)
                    X_batch[x][mask_choice] = MSK_token
                    batch_mask_indices.append(mask_choice)
                for y in range(0,hp_dict["BATCH_SIZE"]):
                    if(y_batch[y][0]):
                        ytrue_dist_bin = [0,1] #true, they are the same genre
                    else:
                        ytrue_dist_bin = [1,0] #false
                    ytrue_bin_batch.append(ytrue_dist_bin) #should give a list BATCHSIZE many same_genre boolean targets

                #convert label lists to pytorch tensors
                ytrue_bin_batch = np.array(ytrue_bin_batch)
                ytrue_multi_batch = np.array(ytrue_multi_batch)
                ytrue_bin_batch = torch.from_numpy(ytrue_bin_batch).float()
                ytrue_multi_batch = torch.from_numpy(ytrue_multi_batch).float()

                #returns predictions for binary class and multiclass, in that order
                ypred_bin_batch,ypred_multi_batch = model(X_batch, batch_mask_indices)

                log.write("For binary classification, predictions are "+str(ypred_bin_batch)+" and true labels are "+str(ytrue_bin_batch)+"\n")
                loss_bin = criterion_bin(ypred_bin_batch, ytrue_bin_batch)
                log.write("The loss in that case was "+str(loss_bin)+"\n")
                log.write("For genre classification, predictions are "+str(ypred_multi_batch)+" and true labels are "+str(ytrue_multi_batch)+"\n")
                loss_multi = criterion_multi(ypred_multi_batch, ytrue_multi_batch)
                log.write("The loss in that case was "+str(loss_multi)+"\n\n")

                #loss = (loss_bin+loss_multi)/2 #as per devlin et al, loss is the average of the two tasks' losses
                loss = loss_bin #toy example for just same-genre task
                #loss = loss_multi
                #acc = get_accuracy(ypred_multi_batch, ytrue_multi_batch, log)
                acc = get_accuracy(ypred_bin_batch, ytrue_bin_batch, log)
                log.write("The total loss this iteration was "+str(loss)+"\n\n")

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
            log.write(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
            #print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')
            #log.write(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')

    log.close()


        #model = torch.load("/Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT/pitchclass/experiment1.pt")
        #model.eval()

