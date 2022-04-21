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

import torch

list_of_tensors = [ torch.randn(3), torch.randn(3), torch.randn(3)]

tensor_of_tensors = torch.stack(list_of_tensors)
print(tensor_of_tensors.shape) #shape (3,3)