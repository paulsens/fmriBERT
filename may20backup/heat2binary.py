import nibabel as nib
import numpy as np
import itertools
import pickle
from Constants import *

path1 = "/Volumes/External/"
#dataset = "opengenre"
dataset = "pitchclass"
path2 = "/preproc/"
path3 ="/STG_masks/"
ant_f = "STG_ant_linearresamp.nii.gz"
post_f = "STG_post_linearresamp.nii.gz"

#### DEPRECATED? #####
#sublist = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665", "1668", "1672", "1678", "1680", "001", "002", "003", "004", "005"]:
sublist = ["1088"]


for sub in sublist:

    subject = "sub-sid00"+sub
    fullpath = path1 + dataset + path2 + subject+"/"+subject + path3

    ant_img = nib.load(fullpath+ant_f)
    ant_data = ant_img.get_fdata()

    post_img = nib.load(fullpath+post_f)
    post_data = post_img.get_fdata()

    leftcount = 0
    rightcount=0

    leftmask = np.zeros((65, 77, 65)) #maybe not actually left brain, but left x-values
    rightmask = np.zeros((65, 77, 65))

    #combine anterior and posterior

    for x in range(0, 65):
        for y in range(0, 77):
            for z in range(0, 65):
                if(post_data[x][y][z]>=threshold or ant_data[x][y][z]>=threshold):
                    if (x<=33): #left side
                        leftmask[x][y][z]=1 #include that voxel in binary mask
                        leftcount += 1
                    else: #right side
                        rightmask[x][y][z]=1 #include that voxel in binary mask
                        rightcount += 1

    print("For sub-sid00"+str(sub)+", leftcount is "+str(leftcount)+" and rightcount is "+str(rightcount))
    left_p = open(fullpath+"STGbinary_left_t"+str(threshold)+".p","wb")
    right_p = open(fullpath+"STGbinary_right_t"+str(threshold)+".p","wb")

    pickle.dump(leftmask,left_p)
    pickle.dump(rightmask,right_p)

    left_p.close()
    right_p.close()

