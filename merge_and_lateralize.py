import nibabel as nib
import numpy as np
import itertools
import pickle
from Constants import *

# CHECK THE SIZE OF YOUR ROI MASK FIRST. THEN DECIDE ON A COOL_DIVIDEND.
COOL_DIVIDEND=420-3 #LEFT STG HAS 410+3 DIMENSIONS, RIGHT STG HAS 428+3 DIMENSIONS, LET'S MEET AT 420 AFTER 3 TOKEN DIMENSIONS ARE ADDED FURTHER DOWNSTREAM

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

    lefthemi = np.zeros((65, 77, 65)) #left x values correspond to left hemisphere according to FSL eyes
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

    #this dictionary adds a layer of abstraction to the following machinery
    hemi_dict = {"left":(lefthemi,leftcount, 0, 33), "right":(righthemi,rightcount, 33,65)}
    for hem in ["left","right"]:
        volume=hemi_dict[hem][0]
        count=hemi_dict[hem][1]
        x_start=hemi_dict[hem][2]
        x_end=hemi_dict[hem][3]

        # If we need to add some voxels slightly below the threshold to get to a nice round dimension
        if (count<COOL_DIVIDEND):
            to_add = COOL_DIVIDEND - count
            candidates=[] #record the to_add many highest probabilities we encounter that aren't already in the mask
            for x in range(x_start, x_end):
                for y in range(0, 77):
                    for z in range(0, 65):
                        bigger = max(post_data[x][y][z], ant_data[x][y][z])
                        if(volume[x][y][z]==0): #if we're not already including that voxel
                            if(len(candidates)<to_add):
                                candidates.append((bigger,x,y,z))
                            else: #only keep the to_add many greatest probabilities
                                candidates = sorted(candidates, key=lambda x: x[0]) #index 0 should have the smallest value
                                if(bigger>candidates[0][0]): #if bigger is greater than the minimum probability in the list
                                    candidates[0]=(bigger,x,y,z) #replace the min with this new one, it will sort next iteration
            print("Our "+str(to_add)+" voxels below threshold with maximal probability are "+str(candidates)+".\n\n")
            for cand in candidates:
                this_x=cand[1]
                this_y=cand[2]
                this_z=cand[3]
                volume[this_x][this_y][this_z]=1 #include that voxel in the mask

        # Need to shave off some valid voxels
        if (count>COOL_DIVIDEND):
            to_shave = count - COOL_DIVIDEND
            candidates=[]
            for x in range(0, 65):
                for y in range(0, 77):
                    for z in range(0, 65):
                        bigger = max(post_data[x][y][z], ant_data[x][y][z])
                        if(volume[x][y][z]==1): #if we're already including that voxel
                            if(len(candidates)<to_shave):
                                candidates.append((bigger,x,y,z))
                            else: #only keep the to_shave many least probabilities
                                candidates = sorted(candidates, key=lambda x: x[0], reverse=True) #index 0 should have the greatest value
                                if(bigger<candidates[0][0]): #if bigger is less likely to be in STG than the maximum probability in the list
                                    candidates[0]=(bigger,x,y,z) #replace the max with this new one, it will sort next iteration
            print("Our "+str(to_shave)+" voxels above threshold with minimal probability are "+str(candidates)+".\n\n")
            for cand in candidates:
                this_x=cand[1]
                this_y=cand[2]
                this_z=cand[3]
                volume[this_x][this_y][this_z]=0 #remove that voxel from the mask

    left_p = open(fullpath+"STGbinary_left_t"+str(threshold)+".p","wb")
    right_p = open(fullpath+"STGbinary_right_t"+str(threshold)+".p","wb")

    pickle.dump(lefthemi,left_p)
    pickle.dump(righthemi,right_p)

    left_p.close()
    right_p.close()

