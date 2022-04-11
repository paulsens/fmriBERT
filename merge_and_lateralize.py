import nibabel as nib
import numpy as np
import itertools
import pickle
from Constants import *


path1 = "/Volumes/External/"
path2 = "/preproc/"
path3 ="/STG_masks/"
ant_f = "STG_ant_linearresamp.nii.gz"
post_f = "STG_post_linearresamp.nii.gz"

# sublist = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665", "1668", "1672", "1678", "1680"]
# dataset = "pitchclass"

sublist = ["1", "2", "3", "4", "5"]
dataset = "opengenre"

#just for testing
#dataset = "pitchclass"
#sublist = ["1088"]


for sub in sublist:

    subject = "sub-sid00"+sub
    fullpath = path1 + dataset + path2 + subject+"/"+subject + path3

    ant_img = nib.load(fullpath+ant_f)
    ant_data = ant_img.get_fdata()
    print("ant data shape is "+str(ant_data.shape))
    post_img = nib.load(fullpath+post_f)
    post_data = post_img.get_fdata()
    print("post data shape is "+str(post_data.shape))
    leftcount = 0
    rightcount=0

    lefthemi = np.zeros((65, 77, 65)) #maybe not actually left brain, but left x-values
    righthemi = np.zeros((65, 77, 65))

    #combine anterior and posterior

    for x in range(0, 65):
        for y in range(0, 77):
            for z in range(0, 65):
                bigger = max(post_data[x][y][z], ant_data[x][y][z])
                if(bigger >= threshold):
                    if (x<=33): #left side
                        #lefthemi[x][y][z]=bigger#include that voxel in left hemisphere
                        lefthemi[x][y][z] = 1
                        leftcount += 1
                    else: #right side
                        #righthemi[x][y][z]=bigger #include that voxel in right hemisphere
                        righthemi[x][y][z] = 1
                        rightcount += 1

    print("For sub-sid00"+str(sub)+", leftcount is "+str(leftcount)+" and rightcount is "+str(rightcount))
    left_p = open(fullpath+"STGbinary_left_t"+str(threshold)+".p","wb")
    right_p = open(fullpath+"STGbinary_right_t"+str(threshold)+".p","wb")

    pickle.dump(lefthemi,left_p)
    pickle.dump(righthemi,right_p)

    left_p.close()
    right_p.close()

