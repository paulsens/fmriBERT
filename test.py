import pickle

path1 = "/Volumes/External/"
path2 = "/preproc/"
path3 ="/STG_masks/"
ant_f = "STG_ant_linearresamp.nii.gz"
post_f = "STG_post_linearresamp.nii.gz"
sub="001"
dataset="opengenre"
subject = "sub-sid" + sub
fullpath = path1 + dataset + path2 + subject + "/" + subject + path3
# threshold="23"
# left_p = open(fullpath + "STGbinary_left_t" + str(threshold) + ".p", "rb")
#
# left_vol = pickle.load(left_p)
#
# count=0
# for x in range(0, 65):
#     for y in range(0, 77):
#         for z in range(0, 65):
#             if left_vol[x][y][z]==1:
#                 count+=1
#
# print("count is "+str(count))
from Constants import *
hemisphere="left"
threshold="23"
for sub in ["001", "002", "003", "004", "005"]:
    iter = 0  # iterations of the next loop, resets per subject
    # opengenre_preproc_path is defined in Constants.py
    subdir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub + "/"
    # load voxel data and labels
    with open(subdir + "STG_allruns" + hemisphere + "_t" + threshold + ".p", "rb") as data_fp:
        all_data = pickle.load(data_fp)
    with open(subdir + "labelindices_allruns.p", "rb") as label_fp:
        all_labels = pickle.load(label_fp)
    print(str(all_labels[0:20]))

    break