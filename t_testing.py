import pickle
import os
from Constants import *
import torch
import copy
from voxel_transformer import Transformer
import random
from random import randint
import torch.nn as nn
import numpy as np
from helpers import get_accuracy
from pitchclass_data import *

from torch.utils.data import DataLoader


#information for finding the right dataset, change manually for different experiments
within_subjects = True
test_copies = 4 #data augmentation for test data
# in a poor turn of events, this is the same name as the variable used for dealing with repetitions in the Test Runs
#  they are totally independent though so hopefully the context will make clear what this is
data_date = "2022-05-04"
verbose = 1
data_count = "0"
task = "binaryonly"
model_count = None
model_path = None
data_path = "dummy"
same_genre_labels = 2  # two possible labels for same genre task, yes or no
num_genres = 10  # from the training set
max_length = 12
seq_len=5
allowed_genres = range(0,10) #all genres for now
voxel_dim = COOL_DIVIDEND + 3  # defined in Constants.py
CLS = [1] + ([0] * (voxel_dim - 1))  # first dimension is reserved for cls_token flag
MSK = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
SEP = [0, 0, 1] + ([0] * (voxel_dim - 3))  # third dimension is reserved for sep_token flag

if env=="local":
    model_path = opengenre_preproc_path+"trained_models/"+task+"/"+data_date+"/"
    data_path = opengenre_preproc_path+"training_data/"+data_date+"/"
if env=="discovery":
    model_path = "/isi/music/auditoryimagery2/seanthesis/pretrainmodels/"+data_date+"/"
    data_path = opengenre_preproc_path+"training_data/"+data_date+"/"
if __name__=="__main__":
    hp_dict = {
        "COOL_DIVIDEND": COOL_DIVIDEND,
        "ATTENTION_HEADS": ATTENTION_HEADS,
        "device": str("cpu"),
        "MSK_flag": 1,
        "CLS_flag": 1,
        "BATCH_SIZE": 1,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": 0.0001,
        # Have to manually set the name of the folder whose training data you want to use, since there will be many
        "data_dir": "2022-05-04",
        # Manually set the hemisphere and iteration number of the dataset you want to use in that folder
        "hemisphere": "left",
        "count": "0",
        # manually set max_seq_length used in data creation, in the input CLS+seq+SEP+seq this is the max length of seq
        "max_sample_length": 5
    }
    # load testsamples and testlabels
    with open(data_path + hp_dict["hemisphere"] + "_testdata" + data_count + ".p", "rb") as samples_fp:
        test_dict = pickle.load(samples_fp)
    with open(data_path + hp_dict["hemisphere"] + "_testlabels" + data_count + ".p", "rb") as labels_fp:
        labels_dict = pickle.load(labels_fp)
    hp_dict["data_path"] = opengenre_preproc_path + "training_data/" + hp_dict["data_dir"] + "/"
    torch.set_default_dtype(torch.float64)

    sublist = test_dict.keys()
    #fill in all the left-samples, then we'll use this is a reference to build the actual test data
    reflist = []
    reflabels = []
    test_X = []
    test_Y = []
    startend_dict = {}
    sub_genre_sample_dict = {}
    sub_id = None

    leftcount = 0
    labelcount = 0
    for sub in sublist:
        genre_sample_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        startend_dict[sub]=[leftcount] #hold the start and end indices for this subject
        samples_per_sub = 0
        lefts = test_dict[sub] #list of left samples
        labels = labels_dict[sub]

        for sample in lefts:
            temp=[]
            temp = temp + copy.deepcopy(sample)
            reflist.append(temp)
            leftcount+=1
        startend_dict[sub].append(leftcount) #current left count is one more than the last index
                                            # but when we use range on these two numbers later it wont be right-inclusive

        for j in range(0, len(labels)):
            label_idx =int(labels[j])
            genre_sample_dict[label_idx].append(j) #get the list of that genre's indices, append this index
            reflabels.append(int(labels[j]))
        #this subject's genre_sample_dict is done, save it in the parent dictionary with sub as key
        sub_genre_sample_dict[sub]=genre_sample_dict

    #reference lists are complete, create the test data based on the value of test_copies

    # for each left-hand sample, create test_copies positive and negative training samples
    for i in range(0, len(reflist)):
        for sub in startend_dict.keys():
            if i>=startend_dict[sub][0] and i<startend_dict[sub][1]:
                sub_id=sub
                break

        #print("sub_id is "+str(sub_id))

        pos_partners = [] # the reference indices with same genre that have already been paired on the right hand side
        neg_partners = [] # the reference indices with different genre that have already been paired on the rhs
        lh_genre = reflabels[i] # genre of left-hand sample
        if lh_genre not in allowed_genres:
            continue #skip the rest of this iteration

        #get a positive partner
        for copy in range(0, test_copies):
            input_pos = [CLS]  # create positive sample, get a new address for each iteration
            input_neg = [CLS]  # create negative sample, get a new address for each iteration

            # create positive training sample, so same genre
            rh_genre = lh_genre
            for j in range(0, seq_len):
                input_pos.append(reflist[i][j])  # fill left hand sample
            input_pos.append(SEP)  # add separator token
            partner = i  # initial condition for loop
            while partner == i:  # find a new partner with the same genre
                if (within_subjects):  # if within_subjects flag is set
                    dict_key = sub_id  # only look at the samples from this subject with this genre
                else:  # pick one of the subjects at random to pair it with, including themself
                    dict_key = randint(1, 5)
                    dict_key = "00" + str(dict_key)
                partner = random.choice(sub_genre_sample_dict[dict_key][
                                            lh_genre])  # all other reference indices of this genre for whichever subject was chosen
                # this seems roundabout, but if within_subjects is not set this is functionally the same as picking one at random from a complete list, while also allowing mostly re-used code if within_subjects is set
                # basically, i did it this way to allow the most overlap in code whether within_subjects is set or not

                if partner in pos_partners:  # did we put this on the right side of this left sample already?
                    partner = i  # if so, keep the loop going

                else:
                    pos_partners.append(partner)  # otherwise put it in the list and we'll exit the loop

            for j in range(0, seq_len):
                input_pos.append(reflist[partner][j])  # fill partner as right hand sample
            pos_label = [1, lh_genre, rh_genre]  # the labels for training, the 1 means same genre
            test_X.append(input_pos)  # add this input to the final list of training inputs
            test_Y.append(pos_label)  # add corresponding positive label vector

            # create negative training sample, so get a different genre
            rh_genre = lh_genre  # initial condition for the loop
            while rh_genre == lh_genre:  # until we get a different one
                rh_genre = random.choice(allowed_genres)  # get a genre label
            for j in range(0, seq_len):
                input_neg.append(reflist[i][j])  # fill left hand sample
            input_neg.append(SEP)  # add separator token

            partner = i  # initial condition for loop
            while partner == i:  # find a new partner with a different genre
                partner = random.choice(sub_genre_sample_dict[dict_key][rh_genre])  # all other reference indices of this genre
                if partner in neg_partners:  # did we put this on the right side of this left sample already?
                    partner = i  # if so, keep the loop going
                else:
                    neg_partners.append(partner)  # otherwise put it in the list and we'll exit the loop
            for j in range(0, seq_len):
                input_neg.append(reflist[partner][j])  # fill partner as right hand sample
            neg_label = [0, lh_genre, rh_genre]  # the labels for training, the 1 means same genre
            test_X.append(input_neg)  # add this input to the final list of training inputs
            test_Y.append(neg_label)  # add corresponding negative label vector


    # now every element of reference_samples should have num_copies many positive and negative partners in training_samples
        #  training_labels[i] is a vector of [{0,1}, lh_genre, rh_genre] for training_sample[i]

    # create the model
    src_pad_sequence = [0] * voxel_dim
    model = Transformer(next_sequence_labels=same_genre_labels, num_genres=num_genres,
                        src_pad_sequence=src_pad_sequence, max_length=max_length, voxel_dim=voxel_dim).to(
        hp_dict["device"])

    criterion_bin = nn.CrossEntropyLoss()
    criterion_multi = nn.CrossEntropyLoss()

    # get MSK indices and set up training targets from label indices
    mask_indices = []
    ytrue_bin = []  # list of batch targets for binary classification task
    ytrue_multi = []  # list of batch targets for multi-classification task
    for x in range(0, len(test_X)):
        mask_choice = randint(1, 10)  # pick a token to mask
        if (mask_choice >= 6):
            mask_choice += 1  # dont want to mask the SEP token at index 6, so 6-10 becomes 7-11
            # each element in the batch has 3 values, same_genre boolean, first half genre, second half genre
            # so if we're masking an element of the second half, the genre decoding label should be that half's genre
            ytrue_multi_idx = test_Y[x][2]
        else:
            ytrue_multi_idx = test_Y[x][1]
        ytrue_dist_multi = np.zeros((10,))  # we want a one-hot probability distrubtion over the 10 genre labels
        ytrue_dist_multi[ytrue_multi_idx] = 1  # set all the probability mass on the true index
        ytrue_multi.append(ytrue_dist_multi)
        test_X[x][mask_choice] = MSK.copy()
        mask_indices.append(mask_choice)
    for y in range(0, len(test_Y)):
        if (test_Y[y][0]):
            ytrue_dist_bin = [0, 1]  # true, they are the same genre
        else:
            ytrue_dist_bin = [1, 0]  # false
        ytrue_bin.append(ytrue_dist_bin)  # should give a list BATCHSIZE many same_genre boolean targets

    # convert label lists to pytorch tensors
    ytrue_bin = np.array(ytrue_bin)
    ytrue_multi = np.array(ytrue_multi)
    ytrue_bin = torch.from_numpy(ytrue_bin).float()
    ytrue_multi = torch.from_numpy(ytrue_multi).float()
    test_X=np.array(test_X)
    test_X = torch.from_numpy(test_X)

    # evaluate the model on the data 226 to 248
    # for weights in os.listdir(model_path):
    #     print(weights)
    #     model.load_state_dict(torch.load(model_path+weights))
    #     with torch.no_grad():
    #         model.eval()
    #         print(test_X)
    #
    #         ypred_bin, ypred_multi = model(test_X, mask_indices)
    #         if task=="binaryonly":
    #             test_loss = criterion_bin(ypred_bin, ytrue_bin)
    #             acc = get_accuracy(ypred_bin, ytrue_bin, None)
    #
    #         elif task=="multionly":
    #             test_loss = criterion_multi(ypred_multi, ytrue_multi)
    #             acc = get_accuracy(ypred_multi, ytrue_multi, None)
    #
    #         else:
    #             # when training both simultaneously, as per devlin et al
    #             test_loss = criterion_bin(ypred_bin, ytrue_bin) + criterion_multi(ypred_multi, ytrue_multi)
    #             test_loss = test_loss/2
    #
    #         print("for model "+str(weights)+" and task "+str(task)+", loss was "+str(test_loss)+" and accuracy was "+str(acc)+"\n")


    #extra testing, delete all below later
    #load samples and labels
    with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_samples"+hp_dict["count"]+".p", "rb") as samples_fp:
        train_X = pickle.load(samples_fp)
    with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_labels"+hp_dict["count"]+".p", "rb") as labels_fp:
        train_Y = pickle.load(labels_fp)
    for weights in os.listdir(model_path):
        print(weights)
        model.load_state_dict(torch.load(model_path+weights))
        model.eval()
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
        #convert to tensors
        train_X = torch.from_numpy(train_X)
        train_Y = torch.from_numpy(train_Y)

        train_data = TrainData(train_X, train_Y) #make the TrainData object
        train_loader = DataLoader(dataset=train_data, batch_size=hp_dict["BATCH_SIZE"], shuffle=True)
        epoch_loss=0
        epoch_acc=0
        MSK_token = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
        MSK_token = np.array(MSK_token)
        MSK_token = torch.from_numpy(MSK_token)
        for X_batch, y_batch in train_loader:
            batch_mask_indices = []
            X_batch, y_batch = X_batch.to(hp_dict["device"]), y_batch.to(hp_dict["device"])
            ytrue_bin_batch = []  # list of batch targets for binary classification task
            ytrue_multi_batch = []  # list of batch targets for multi-classification task
            for x in range(0, hp_dict["BATCH_SIZE"]):
                mask_choice = randint(1, 10)  # pick a token to mask
                if (mask_choice >= 6):
                    mask_choice += 1  # dont want to mask the SEP token at index 6, so 6-10 becomes 7-11
                    # each element in the batch has 3 values, same_genre boolean, first half genre, second half genre
                    # so if we're masking an element of the second half, the genre decoding label should be that half's genre
                    ytrue_multi_idx = y_batch[x][2]
                else:
                    ytrue_multi_idx = y_batch[x][1]
                ytrue_dist_multi = np.zeros((10,))  # we want a one-hot probability distrubtion over the 10 genre labels
                ytrue_dist_multi[ytrue_multi_idx] = 1  # set all the probability mass on the true index
                ytrue_multi_batch.append(ytrue_dist_multi)
                X_batch[x][mask_choice] = MSK_token
                batch_mask_indices.append(mask_choice)
            for y in range(0, hp_dict["BATCH_SIZE"]):
                if (y_batch[y][0]):
                    ytrue_dist_bin = [0, 1]  # true, they are the same genre
                else:
                    ytrue_dist_bin = [1, 0]  # false
                ytrue_bin_batch.append(ytrue_dist_bin)  # should give a list BATCHSIZE many same_genre boolean targets

            # convert label lists to pytorch tensors
            ytrue_bin_batch = np.array(ytrue_bin_batch)
            ytrue_multi_batch = np.array(ytrue_multi_batch)
            ytrue_bin_batch = torch.from_numpy(ytrue_bin_batch).float()
            ytrue_multi_batch = torch.from_numpy(ytrue_multi_batch).float()

            # returns predictions for binary class and multiclass, in that order
            ypred_bin_batch, ypred_multi_batch = model(X_batch, batch_mask_indices)

            loss_bin = criterion_bin(ypred_bin_batch, ytrue_bin_batch)
            loss_multi = criterion_multi(ypred_multi_batch, ytrue_multi_batch)

            # loss = (loss_bin+loss_multi)/2 #as per devlin et al, loss is the average of the two tasks' losses
            loss = loss_bin  # toy example for just same-genre task
            # loss = loss_multi
            # acc = get_accuracy(ypred_multi_batch, ytrue_multi_batch, log)
            acc = get_accuracy(ypred_bin_batch, ytrue_bin_batch, None)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print(f'Epoch 1: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
