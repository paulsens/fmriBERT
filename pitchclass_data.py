import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset
from random import randint

class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

def load_subject(subjectID):
    datapath = "/Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT/pitchclass/samples/"

    data_fp = open(datapath+subjectID+".p","rb")
    samples = pickle.load(data_fp, encoding="bytes")
    print(len(samples[0]))
    data_fp.close()

    return samples

def make_padded_samples(samples):
    TIMESTEPS = len(samples)
    voxel_dim = len(samples[0])
    padded_samples = []
    print(samples[0])
    for t in range(0, TIMESTEPS):
        temp = [0, 0, 0] #dimensions for CLS, MSK, and SEP flags
        empty_pad_ctr = 0
        while ((len(temp)+voxel_dim) % 4) !=0:
            temp += [0]
            empty_pad_ctr += 1
        temp = temp + samples[t].tolist()

        padded_samples.append(temp)

    return padded_samples

#pass in a timeseries of voxel data for a single subject
# returns a inputs and labels for next sequence prediction task
def make_NS_data(samples, seq_len, TIMESTEPS, voxel_dim, CLS_token, SEP_token):
    #possible starting points are 0 through TIMESTEPS - (seq_len*2)
    X_data = []
    y_data = []
    FAKE = [0,0,0]+[33]*(voxel_dim-3)
    goodfake = [0, 0, 0]+[1]*(voxel_dim-3)
    for i in range(0, TIMESTEPS-(seq_len*2)):
        temp = [CLS_token]
        for j in range(0, seq_len): #append current sequence
            temp.append(samples[i+j])

        temp.append(SEP_token)
        flip = randint(0,1)
        if(flip): #append the actual next sequence
            for k in range(0, seq_len):
                #temp.append(samples[i+seq_len+k]) #start appending after the current sequence ended
                temp.append(goodfake)
            y_data.append([0,1])
        else: #append a fake next sequence
            for k in range(0, seq_len):
                temp.append(FAKE)
            y_data.append([0,1])
        X_data.append(temp)


    X_data = torch.FloatTensor(X_data)
    y_data = torch.FloatTensor(y_data)

    print("X_data has shape "+str(X_data.shape))
    print("y_data has shape "+str(y_data.shape))

    return X_data, y_data

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    # print("y_pred is "+str(y_pred))
    # print("y_pred_tag is" + str(y_pred_tag))
    # print("y_test is "+str(y_test))
    correct_results_sum = 0
    for i in range(0, len(y_pred)):
        if y_pred[i].tolist() == y_test[i].tolist():
            correct_results_sum+=1
            print("yes, "+str(y_pred[i])+" equals "+str(y_test[i]))
    print("correct results sum is "+str(correct_results_sum))
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.FloatTensor([acc,])
    acc = torch.round(acc*100)

    return acc


