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

val_flag=0
random.seed(5)
#datachoice = "old"
datachoice="new"
#modelchoice="old"
modelchoce="new"

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
        today = datetime.date.today()
        now = datetime.datetime.now()
        ##############################  SET PARAMETERS  ##############################
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #dictionary of hyperparameters, eventually should probably come from command line
        hp_dict={
            "task":"binaryonly",
            "COOL_DIVIDEND" : COOL_DIVIDEND,
            "ATTENTION_HEADS" : ATTENTION_HEADS,
            "device" : str(device),
            "MSK_flag" : 1,
            "CLS_flag" : 1,
            "BATCH_SIZE" : 1,
            "EPOCHS" : EPOCHS,
            "LEARNING_RATE" : 0.00005,
            #Have to manually set the name of the folder whose training data you want to use, since there will be many
            "data_dir" : "seeded/2022-05-19",
            #Manually set the hemisphere and iteration number of the dataset you want to use in that folder
            "hemisphere": "left",
            "count" : str(thiscount),
            #manually set max_seq_length used in data creation, in the input CLS+seq+SEP+seq this is the max length of seq
            "max_sample_length":5
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
        logfile = today_dir + "seededpretrain_log_"+str(logcount)+".txt"
        while (os.path.exists(logfile)):
            logcount+=1
            logfile = today_dir + "pretrain_log_" + str(logcount) + ".txt"
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
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_seededtrainsamplesorig"+hp_dict["count"]+".p", "rb") as samples_fp:
            train_X = pickle.load(samples_fp)
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_seededtrainlabelsorig"+hp_dict["count"]+".p", "rb") as labels_fp:
            train_Y = pickle.load(labels_fp)

        #load valsamples and vallabels
        if(val_flag==0):
            val_X=None
            val_Y=None
        else:
            with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_valsamples"+hp_dict["count"]+".p", "rb") as samples_fp:
                val_X = pickle.load(samples_fp)
            with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_vallabels"+hp_dict["count"]+".p", "rb") as labels_fp:
                val_Y = pickle.load(labels_fp)

        #load samples with old path format


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

        model = Transformer(next_sequence_labels=same_genre_labels, num_genres=num_genres, src_pad_sequence=src_pad_sequence, max_length=max_length, voxel_dim=voxel_dim).to(hp_dict["device"])
        model.to(hp_dict["device"])

        criterion_bin = nn.CrossEntropyLoss()
        criterion_multi = nn.CrossEntropyLoss()
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
                    #X_batch[x][mask_choice] = torch.clone(MSK_token)
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
                #batch_mask_indices = np.array(batch_mask_indices)

                ytrue_bin_batch = torch.from_numpy(ytrue_bin_batch).float()
                ytrue_multi_batch = torch.from_numpy(ytrue_multi_batch).float()
                #batch_mask_indices = torch.from_numpy(batch_mask_indices).float()

                #send this stuff to device
                ytrue_bin_batch.to(hp_dict["device"])
                ytrue_multi_batch.to(hp_dict["device"])
                #batch_mask_indices.to(hp_dict["device"])

                #returns predictions for binary class and multiclass, in that order
                ypred_bin_batch,ypred_multi_batch = model(X_batch, batch_mask_indices)

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
                    acc = get_accuracy(ypred_multi_batch, ytrue_multi_batch, log)
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
                    for X_val, y_val in val_loader:
                        model.eval() #set the model to evaluation mode

                        val_mask_indices = []
                        X_batch, y_batch = X_batch.to(hp_dict["device"]), y_batch.to(hp_dict["device"])

                        ytrue_bin_batch = []  # list of batch targets for binary classification task
                        ytrue_multi_batch = []  # list of batch targets for multi-classification task
                        for x in range(0, hp_dict["BATCH_SIZE"]):
                            mask_choice = randint(1, 10)  # pick a token to mask
                            if (mask_choice >= 6):
                                mask_choice += 1  # dont want to mask the SEP token at index 6, so 6-10 becomes 7-11
                                # each element in the batch has 3 values, same_genre boolean, first half genre, second half genre
                                # so if we're masking an element of the second half, the genre decoding label should be that half's genre
                                ytrue_multi_idx = y_val[x][2]
                            else:
                                ytrue_multi_idx = y_val[x][1]
                            ytrue_dist_multi = np.zeros(
                                (10,))  # we want a one-hot probability distrubtion over the 10 genre labels
                            ytrue_dist_multi[ytrue_multi_idx] = 1  # set all the probability mass on the true index
                            ytrue_multi_batch.append(ytrue_dist_multi)
                            X_val[x][mask_choice] = torch.clone(MSK_token)
                            val_mask_indices.append(mask_choice)
                        for y in range(0, hp_dict["BATCH_SIZE"]):
                            if (y_val[y][0]):
                                ytrue_dist_bin = [0, 1]  # true, they are the same genre
                            else:
                                ytrue_dist_bin = [1, 0]  # false
                            ytrue_bin_batch.append(
                                ytrue_dist_bin)  # should give a list BATCHSIZE many same_genre boolean targets

                        # convert label lists to pytorch tensors
                        ytrue_bin_batch = np.array(ytrue_bin_batch)
                        ytrue_multi_batch = np.array(ytrue_multi_batch)
                        ytrue_bin_batch = torch.from_numpy(ytrue_bin_batch).float()
                        ytrue_multi_batch = torch.from_numpy(ytrue_multi_batch).float()

                        # returns predictions for binary class and multiclass, in that order
                        ypred_bin_batch, ypred_multi_batch = model(X_val, val_mask_indices)

                        # log.write("For binary classification, predictions are "+str(ypred_bin_batch)+" and true labels are "+str(ytrue_bin_batch)+"\n")
                        loss_bin = criterion_bin(ypred_bin_batch, ytrue_bin_batch)
                        # log.write("The loss in that case was "+str(loss_bin)+"\n")
                        # log.write("For genre classification, predictions are "+str(ypred_multi_batch)+" and true labels are "+str(ytrue_multi_batch)+"\n")
                        loss_multi = criterion_multi(ypred_multi_batch, ytrue_multi_batch)
                        # log.write("The loss in that case was "+str(loss_multi)+"\n\n")

                        if hp_dict["task"] == "binaryonly":
                            loss = loss_bin  # toy example for just same-genre task
                            acc = get_accuracy(ypred_bin_batch, ytrue_bin_batch, log)
                        elif hp_dict["task"] == "multionly":
                            loss = loss_multi
                            acc = get_accuracy(ypred_multi_batch, ytrue_multi_batch, log)
                        else:
                            loss = (
                                               loss_bin + loss_multi) / 2  # as per devlin et al, loss is the average of the two tasks' losses
                            acc = 0
                        # log.write("The total loss this iteration was "+str(loss)+"\n\n")

                        val_loss += loss.item()
                        val_acc += acc.item()

            # training is done, load in the test data
            #  recall that the mask tokens are already applied to this data for consistent results
            # this_dir = hp_dict["data_path"] + hp_dict["hemisphere"] + "_"
            # with open(this_dir + "testX" + hp_dict["count"] + ".p", "rb") as testX_fp:
            #     test_X = pickle.load(testX_fp)
            # with open(this_dir + "testY_bin" + hp_dict["count"] + ".p", "rb") as testY_bin_fp:
            #     testY_bin = pickle.load(testY_bin_fp)
            # with open(this_dir + "testY_multi" + hp_dict["count"] + ".p", "rb") as testY_multi_fp:
            #     testY_multi = pickle.load(testY_multi_fp)
            # with open(this_dir + "test_maskidxs" + hp_dict["count"] + ".p", "rb") as test_maskidxs_fp:
            #     test_maskidxs = pickle.load(test_maskidxs_fp)
            #
            # with torch.no_grad():
            #     model.eval()
            #
            #     ypred_bin, ypred_multi = model(test_X, test_maskidxs)
            #     if hp_dict["task"] == "binaryonly":
            #         test_loss = criterion_bin(ypred_bin, testY_bin)
            #         acc = get_accuracy(ypred_bin, testY_bin, None)
            #
            #     elif hp_dict["task"] == "multionly":
            #         test_loss = criterion_multi(ypred_multi, testY_multi)
            #         acc = get_accuracy(ypred_multi, testY_multi, None)
            #
            #     else:
            #         # when training both simultaneously, as per devlin et al
            #         test_loss = criterion_bin(ypred_bin, testY_bin) + criterion_multi(ypred_multi, testY_multi)
            #         test_loss = test_loss / 2
            #
            #     print("for test data, loss was " + str(test_loss) + " and accuracy was " + str(acc) + "\n")
            #     log.write("for test data, loss was " + str(test_loss) + " and accuracy was " + str(acc) + "\n")


            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

            log.write(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
            if val_X is not None:
                print(f'Validation: | Loss: {val_loss/len(val_loader):.5f} | Acc: {val_acc/len(val_loader):.3f}')
                log.write(f'Validation: | Loss: {val_loss/len(val_loader):.5f} | Acc: {val_acc/len(val_loader):.3f}')

            #print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')
            #log.write(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')


        # training is done, load in the test data
        #  recall that the mask tokens are already applied to this data for consistent results
        # this_dir = hp_dict["data_path"]+hp_dict["hemisphere"]+"_"
        # with open(this_dir+"testX"+hp_dict["count"]+".p", "rb") as testX_fp:
        #     test_X = pickle.load(testX_fp)
        # with open(this_dir+"testY_bin"+hp_dict["count"]+".p", "rb") as testY_bin_fp:
        #     testY_bin = pickle.load(testY_bin_fp)
        # with open(this_dir+"testY_multi"+hp_dict["count"]+".p", "rb") as testY_multi_fp:
        #     testY_multi = pickle.load(testY_multi_fp)
        # with open(this_dir+"test_maskidxs"+hp_dict["count"]+".p", "rb") as test_maskidxs_fp:
        #     test_maskidxs = pickle.load(test_maskidxs_fp)
        #
        # with torch.no_grad():
        #     model.eval()
        #
        #     ypred_bin, ypred_multi = model(test_X, test_maskidxs)
        #     if hp_dict["task"]=="binaryonly":
        #         test_loss = criterion_bin(ypred_bin, testY_bin)
        #         acc = get_accuracy(ypred_bin, testY_bin, None)
        #
        #     elif hp_dict["task"]=="multionly":
        #         test_loss = criterion_multi(ypred_multi, testY_multi)
        #         acc = get_accuracy(ypred_multi, testY_multi, None)
        #
        #     else:
        #         # when training both simultaneously, as per devlin et al
        #         test_loss = criterion_bin(ypred_bin, testY_bin) + criterion_multi(ypred_multi, testY_multi)
        #         test_loss = test_loss/2
        #
        #     print("for test data, loss was "+str(test_loss)+" and accuracy was "+str(acc)+"\n")
        #     log.write("for test data, loss was "+str(test_loss)+" and accuracy was "+str(acc)+"\n")
        # # save the weights of the model, not sure if this actually works
        model_path = opengenre_preproc_path+"trained_models/"+hp_dict["task"]+"/seededorig_"+str(thiscount)+".pt"
        torch.save(model.state_dict(),model_path)

    log.close()
