from random import randint
import numpy as np
from helpers import *
from Constants import *
import sys
import os
import datetime
import random
import pickle
hemisphere="left"
holdout=0
count=0
this_dir = opengenre_preproc_path + "training_data/seeded/2022-05-19/"

old_samples = this_dir + str(hemisphere) + "_seededtrainsamplesorig" + str(count) + ".p"
old_labels = this_dir + str(hemisphere) + "_seededtrainlabelsorig" + str(count) + ".p"

new_samples = this_dir + str(hemisphere) + "_seededtrainsamplesnew" + str(holdout) + ".p"
new_labels = this_dir + str(hemisphere) + "_seededtrainlabelsnew" + str(holdout) + ".p"
# save training data and labels
with open(old_samples, "rb") as oldsamples_fp:
    osamples = np.array(pickle.load(oldsamples_fp))
with open(old_labels, "rb") as oldlabels_fp:
    olabels = np.array(pickle.load(oldlabels_fp))
with open(new_samples, "rb") as newsamples_fp:
    nsamples = np.array(pickle.load(newsamples_fp))
with open(new_labels, "rb") as newlabels_fp:
    nlabels = np.array(pickle.load(newlabels_fp))

print("lengths in order:  \n"
      +str(osamples.shape)+" \n"
      +str(olabels.shape)+"\n"
      +str(nsamples.shape)+"\n"
      +str(nlabels.shape)+"\n")

print(osamples[0])
print(olabels[0])
print(nsamples[0])
print(nlabels[0])