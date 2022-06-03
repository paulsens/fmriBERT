import numpy as np
import torch
import random
import pickle

datadir="/Volumes/External/opengenre/preproc/training_data/"
modeldir = "/Volumes/External/opengenre/preproc/trained_models/june2/"
goodsamples=datadir+"2022-05-02/left_samples0.p"
goodlabels=datadir+"2022-05-02/left_labels0.p"
badsamples=datadir+"seeded/2022-05-19/left_seededtrainsamplesorig0.p"
badlabels=datadir+"seeded/2022-05-19/left_seededtrainlabelsorig0.p"

with open(goodsamples,"rb") as gs:
    gsamples = pickle.load(gs)
with open(goodlabels,"rb") as gl:
    glabels = pickle.load(gl)
with open(badsamples,"rb") as bs:
    bsamples = pickle.load(bs)
with open(badlabels,"rb") as bl:
    blabels=pickle.load(bl)


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

def labelstats(thelist, type):
    thelist=np.array(thelist)
    mymat=np.zeros((10,10))
    for row in thelist:
        g1=int(row[1])
        g2=int(row[2])
        mymat[g1][g2]+=1
    print("For "+str(type)+" labels, collision matrix is")
    print(str(mymat))

random.seed(3)
#peekdata(glabels, "good", 20)
labelstats(glabels,"good")
print("\n\n")
random.seed(3)
#peekdata(blabels, "bad", 20)
labelstats(blabels,"bad")