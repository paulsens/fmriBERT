import pickle
from Constants import targets_dir
with open(targets_dir+"sub_5gram_dict.p", "rb") as sub_fp:
    sub_5gram_dict = pickle.load(sub_fp)

with open(targets_dir+"idx_5gram_dict.p", "rb") as idx_fp:
    idx_5gram_dict = pickle.load(idx_fp)

vivid_idx=idx_5gram_dict["vividness"]
gof_idx=idx_5gram_dict["GoF"]

viv_list =[0, 0, 0, 0, 0]
gof_list = [0, 0, 0, 0, 0]

#----------- Just some code for counting the frequency of vividness and GoF ratings
# for sub in sub_5gram_dict.keys():
#     for cycle in sub_5gram_dict[sub]: #this is a list of tuples, get the index of desiderata using idx_5gram_dict
#         viv=cycle[vivid_idx]
#         gof=cycle[gof_idx]
#
#
#         viv_list[viv]+=1
#         gof_list[gof]+=1




