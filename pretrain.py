import torch
import torch.nn as nn
from random import randint
import numpy as np
from voxel_transformer import *
from pitchclass_data import *
from torch.utils.data import DataLoader
import torch.optim as optim
from Constants import *

if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):

        ##############################  SET PARAMETERS  ##############################
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        MSK_flag = 1
        CLS_flag = 1
        BATCH_SIZE = 1
        EPOCHS = 2
        LEARNING_RATE = 0.001
        #Have to manually set the name of the folder whose training data you want to use, since there will be many
        data_dir = "2022-04-15-13-08-29.170995"
        #manually set max_seq_length used in data creation, in the input CLS+seq+SEP+seq this is the max length of seq
        max_sample_length=5
        data_path = opengenre_preproc_path + "training_data/" + data_dir + "/"
        torch.set_default_dtype(torch.float64)

        #load samples and labels
        with open(data_path + "samples.p", "rb") as samples_fp:
            train_X = pickle.load(samples_fp)
        with open(data_path + "labels.p", "rb") as labels_fp:
            train_Y = pickle.load(labels_fp)
        #train_X has shape (timesteps, max_length, voxel_dim)
        num_samples = len(train_X)
        max_length = len(train_X[0]) #should be max_length*2 + 2
        assert (max_length == (max_sample_length*2 +2))

        voxel_dim = len(train_X[0][0])

        #convert to numpy arrays
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
        #convert to tensors
        train_X = torch.from_numpy(train_X)
        train_Y = torch.from_numpy(train_Y)

        train_data = TrainData(train_X, train_Y) #make the TrainData object
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True) #make the DataLoader object
        print("voxel dim is "+str(voxel_dim))
        MSK_token = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
        MSK_token = np.array(MSK_token)
        MSK_token = torch.from_numpy(MSK_token)

        same_genre_labels = 2 #two possible labels for same genre task, yes or no
        num_genres = 10 #from the training set

        src_pad_sequence = [0]*voxel_dim

        cross_entropy_loss = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss(reduction='batchmean')


        model = Transformer(next_sequence_labels=same_genre_labels, num_genres=num_genres, src_pad_sequence=src_pad_sequence, max_length=max_length, voxel_dim=voxel_dim).to(device)
        model.to(device)

        criterion_bin = nn.BCEWithLogitsLoss()
        criterion_multi = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        model.train() #sets model status, doesn't actually start training
        for e in range(1, EPOCHS+1):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in train_loader:
                batch_mask_indices = []
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                ytrue_bin_batch = [] #list of batch targets for binary classification task
                ytrue_multi_batch = [] #list of batch targets for multi-classification task
                optimizer.zero_grad()
                for x in range(0,BATCH_SIZE):
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
                for y in range(0,BATCH_SIZE):
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

                loss_bin = criterion_bin(ypred_bin_batch, ytrue_bin_batch)
                loss_multi = criterion_multi(ypred_multi_batch, ytrue_multi_batch)

                loss = (loss_bin+loss_multi)/2 #as per devlin et al, loss is the average of the two tasks' losses
                #acc = binary_acc(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                #epoch_acc += acc.item()

            #print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')

        #model = torch.load("/Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT/pitchclass/experiment1.pt")
        #model.eval()

