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
from helpers import mask_flatten_combine_opengenre

old_samples = this_dir + str(hemisphere) + "_seededtrainsamplesorig" + str(count) + ".p"
old_labels = this_dir + str(hemisphere) + "_seededtrainlabelsorig" + str(count) + ".p"

new_samples = this_dir + str(hemisphere) + "_seededtrainsamplesnew" + str(holdout) + ".p"
new_labels = this_dir + str(hemisphere) + "_seededtrainlabelsnew" + str(holdout) + ".p"
# save training data and labels
# with open(old_samples, "rb") as oldsamples_fp:
#     osamples = np.array(pickle.load(oldsamples_fp))
# with open(old_labels, "rb") as oldlabels_fp:
#     olabels = np.array(pickle.load(oldlabels_fp))
# with open(new_samples, "rb") as newsamples_fp:
#     nsamples = np.array(pickle.load(newsamples_fp))
# with open(new_labels, "rb") as newlabels_fp:
#     nlabels = np.array(pickle.load(newlabels_fp))
idxs = [1, 3, 4, 5, 6, 7]
poop = random.sample(idxs,k=1)
print(poop)