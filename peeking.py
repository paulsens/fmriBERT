import numpy as np
import torch
import random
import pickle
from helpers import get_accuracy

datadir="/Volumes/External/opengenre/preproc/training_data/"
modeldir = "/Volumes/External/opengenre/preproc/trained_models/june2/"
goodsamples=datadir+"2022-05-02/left_samples0.p"
goodlabels=datadir+"2022-05-02/left_labels0.p"
badsamples=datadir+"seeded/2022-05-19/left_seededtrainsamplesorig0.p"
badlabels=datadir+"seeded/2022-05-19/left_seededtrainlabelsorig0.p"
savedmodel="/Volumes/External/opengenre/preproc/trained_models/may2data/may2modelgooddata.pt"
voxel_dim=420

with open(goodsamples,"rb") as gs:
    gsamples = pickle.load(gs)
with open(goodlabels,"rb") as gl:
    glabels = pickle.load(gl)
with open(badsamples,"rb") as bs:
    bsamples = pickle.load(bs)
with open(badlabels,"rb") as bl:
    blabels=pickle.load(bl)
# with open(savedmodel,"rb") as mod:
#     model=torch.load(mod)

MSK_token = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
MSK_token = np.array(MSK_token)
MSK_token = torch.from_numpy(MSK_token)

def try_dataset(modeldir, data, labels, task):
    model=torch.load(modeldir)
    model.eval()
    data=np.array(data)
    data=torch.from_numpy(data)
    print("data tensor has shape "+str(data.shape))
    running_acc=0
    for sample in range(0, len(data)):
        if(sample==0):
            print("length is "+str(len(data[sample])))
        idx=random.randint(1,10)
        if idx>5:
            idx+=1
        data[sample][idx]=MSK_token
        length=len(data[sample])
        length2=len(data[sample][0])
        temp=data[sample]
        temp = torch.reshape(temp,(1,length,length2))
        #print("after reshape, temp has shape "+str(temp.shape))
        bin_pred, multi_pred=model(temp, [idx])
        bin_true=labels[sample][0]
        if(idx>5):
            multi_true=labels[sample][2]
        else:
            multi_true=labels[sample][1]
        if(bin_true):
            bin_true_dist=[0,1]
        else:
            bin_true_dist=[1,0]
        multi_true_dist = np.zeros((10,))  # we want a one-hot probability distrubtion over the 10 genre labels
        multi_true_dist[multi_true]=1

        bin_true_dist=np.array(bin_true_dist)
        bin_true_dist=torch.from_numpy(bin_true_dist)
        multi_true_dist=np.array(multi_true_dist)
        multi_true_dist=torch.from_numpy(multi_true_dist)

        if(task=="binaryonly"):
            bin_true_dist=torch.reshape(bin_true_dist, (1,len(bin_true_dist)))
            acc = get_accuracy(bin_pred, bin_true_dist, None)
        elif(task=="multionly"):
            acc = get_accuracy(multi_pred, multi_true_dist, None)
        running_acc+=acc
    final_acc=running_acc/(len(data))
    print("final_acc is "+str(final_acc))


def peekdata(thelist, type, num_peeks):
    thearray=np.array(thelist)
    dim = len(thearray.shape)
    samples = thearray.shape[0]
    if dim==2: # if it's labels
        print("Peeking at "+str(type)+" labels:\n")
        choice_a=random.choices(range(0,samples),k=num_peeks)
        print("choice is "+str(choice_a)+"\n")
        for c in choice_a:
            print(str(thearray[c])+"\n")
    else:
        choice = random.randint(0, samples)
        print("Peeking  at "+str(type)+" samples:\n")
        print(str(thearray[choice]))


def labelstats(thelist, type):
    thelist=np.array(thelist)
    mymat=np.zeros((10,10))
    for row in thelist:
        g1=int(row[1])
        g2=int(row[2])
        mymat[g1][g2]+=1
    print("For "+str(type)+" labels, collision matrix is")
    print(str(mymat))

try_dataset(savedmodel, gsamples, glabels, "binaryonly")
#labelstats(glabels, "good")
#peekdata(gsamples, "good", 1)
print("\n")
#labelstats(blabels, "bad")
#peekdata(bsamples, "bad", 1)