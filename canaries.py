import random
import pickle
import copy
import numpy as np
import torch

# samples is the total number of inputs that mask tokens need to be applied to, i.e the size of the training set
 # seq_length is the length of input samples including CLS and SEP
 # seed is the RNG seed
 # within_subject is either 0 if replacing among subject, or is the number of subjects if replacing within subject
def make_mask_indices(samples, seq_length, seed, canary_path, within_subject=0):
    msk_register = {} #the key is always the index in the training data, so [0,samples-1]
                    # the value is a colon-segmented string
                    # 1:7-m: indicates there is one masked token, index 7 was replaced by MSK
                    # 0 indicates no replaced tokens
                    # 2:3-r-x-y:8-m means there are two, index 3 was replaced by index y in sample x
                                                        # and index 8 was replaced by MSK
    if(within_subject>0):
        samples_per_subject=samples//within_subject
    else:
        samples_per_subject=None
    random.seed(seed)
    idx_range = list(range(1, seq_length//2)) + list(range(seq_length//2+1, seq_length)) #skips over the positions of CLS and SEG
    for i in range(0, samples):
        if(within_subject>0):
            i_floor = i//samples_per_subject
            i_floor = i_floor*samples_per_subject
            i_ceiling = i_floor + samples_per_subject # 1 greater than the maximum i value for that subject
        else:
            i_floor = 0
            i_ceiling = samples
        i_choice, r_choice, i2_choice, r2_choice=None, None, None, None
        istr = ""
        first_str=""
        second_str=""
        r_count = 0 # number of replacements applied so far, doesn't increase in 10% case where it's unaffected
        flip = random.choice(range(1,101)) #are we doing two masks on this sample or one?

        if flip <= 50: # only one token being replaced
            mask_n = 1
            i_choice = random.choice(idx_range)
            r_choice = random.randint(1, 101)  # 1-10 inclusive means leave it alone, 11-20 means random replacement, 21+ means MSK
        else: # we're attempting to replace two tokens instead of one
            mask_n = 2 # doesn't account for 10% chance to leave it unaffected
            choices_l = random.sample(idx_range,k=2) # to make sure ichoice and i2choice aren't the same
            i_choice=choices_l[0]
            i2_choice = choices_l[1]

            r_choice = random.randint(1, 101)  # 1-10 inclusive means leave it alone
            r2_choice = random.randint(1, 101)  # 1-10 inclusive means leave it alone

        if (r_choice>10): #if it's not left unaffected
            r_count += 1
            first_str = get_half_string(i, samples, idx_range, i_choice, r_choice, i_floor, i_ceiling)

        if mask_n==2: #if we need a second one
            if (r2_choice>10):
                r_count += 1
                second_str = get_half_string(i, samples, idx_range, i2_choice, r2_choice, i_floor, i_ceiling)

        istr = str(r_count)+":"+first_str+":"+second_str
        msk_register[i]=istr

    return msk_register

def apply_mask_register(canary_path, register_str, data_path):
    #prelim stuff
    with open(data_path,"rb") as data_fp:
        dataset = pickle.load(data_fp)
    voxel_dim=len(dataset[0][0])
    MSK = [0, 1] + ([0] * (voxel_dim - 2))
    with open(canary_path+register_str, "rb") as reg_fp:
        reg = pickle.load(reg_fp)

    #start masking/replacing
    for i in range(0, len(dataset)): # i is the index into dataset as well as the key into the mask register
        mask_str = reg[i]
        mask_l = mask_str.split(":")
        if mask_l[0]=="0": #if nothing is getting replaced on this value of i, move on
            continue
        # if we get here, mask_l[1] is not empty
        first_l = mask_l[1].split("-")
        if len(first_l)==2: # something like "7-m" split into ["7","m"]
            idx=int(first_l[0])
            dataset[i][idx]=copy.deepcopy(MSK)
        else: # something like 4-r-17-9
            idx=int(first_l[0])
            r_idx=int(first_l[2])
            r_slot=int(first_l[3])
            dataset[i][idx]=copy.deepcopy(dataset[r_idx][r_slot]) # direct replacement

        # now do it again if we need to
        if(mask_l[0]=="2"):
            second_l = mask_l[2].split("-")
            if len(second_l) == 2:  # something like "7-m" split into ["7","m"]
                idx = int(second_l[0])
                dataset[i][idx] = copy.deepcopy(MSK)
            else:  # something like 4-r-17-9
                idx = int(second_l[0])
                r_idx = int(second_l[2])
                r_slot = int(second_l[3])
                dataset[i][idx] = copy.deepcopy(dataset[r_idx][r_slot])  # direct replacement

    # looped through the whole dataset, masking process is done
    return dataset


def get_half_string(i, samples, idx_range, i_choice, r_choice, i_floor, i_ceiling):
    the_str=""
    if (r_choice < 21):  # 10-20 inclusive means random replacement
        r_idx = i
        while r_idx == i:
            r_idx = random.choice(range(i_floor, i_ceiling))
        r_slot = random.choice(idx_range)  # which element of sample r_idx will replace element i_choice in sample i
        the_str = str(i_choice) + "-r-" + str(r_idx) + "-" + str(r_slot)
    else:  # >20 means standard mask replacement
        the_str = str(i_choice) + "-m"

    return the_str

# note that this is making the list of left-samples, so seq_len is only one half without CLS or SEP
def make_gaussian_data(n_samples, voxel_dim, seq_len, seed, means, sdev):
    samples_per_mean=n_samples//len(means)
    random.seed(seed)
    TIMESTEPS = n_samples
    mydataset = []
    mylabels = []
    for i in range(0, TIMESTEPS):
        # start a new list
        temp = np.zeros((seq_len, voxel_dim))
        sample = torch.normal(mean, sdev, size=(5, voxel_dim-3)) #dont include token dimensions in these vectors
        # print("sample is "+str(sample))
        for j in range(0, 5):
            for k in range(3, voxel_dim):
                temp[j][k] = sample[j][k - 3]
        mydataset.append(temp.tolist())
    return mydataset

def make_opengenre_data():
    return None

def write_metadata(vars_dict, file_str, path_str):
    with open(path_str+file_str+"_metadata.txt", "w") as md_fp:
        for key in vars_dict.keys():
            md_fp.write(str(key)+" : "+str(vars_dict[key])+"\n")

def write_file(data, file_str, path_str):
    with open(path_str+file_str+".p", "wb") as data_fp:
        pickle.dump(data, data_fp)


# MAKE A CANARY OF MASK INDICES

# vars_dict= {"samples":10800,
#             "epochs":1,
#             "seq_length":12,
#             "seed":3,
#             "canary_path":"/Volumes/External/opengenre/preproc/canaries/",
# }
# file_str="mask_indices_0"
# indices = make_mask_indices(vars_dict["samples"], vars_dict["seq_length"], vars_dict["seed"], vars_dict["canary_path"], within_subject=5)
# write_file(indices, file_str, vars_dict["canary_path"])
# write_metadata(vars_dict, file_str, vars_dict["canary_path"])

# MAKE A CANARY OF GAUSSIAN DATA
vars_dict= {"samples":5400,
            "epochs":1,
            "seq_length":5,
            "seed":3,
            "means":[0,1, 2, 3, 4, 5, 6, 7, 8, 9],
            "sdev":1,
            "voxel_dim":420,
            "canary_path":"/Volumes/External/opengenre/preproc/canaries/",
}
gdata=make_gaussian_data(vars_dict["samples"], vars_dict["voxel_dim"], vars_dict["seq_length"], vars_dict["seed"], vars_dict["means"], vars_dict["sdev"])
file_str="gaussian_data_1"
write_file(gdata,file_str, vars_dict["canary_path"])
write_metadata(vars_dict, file_str, vars_dict["canary_path"])

# MAKE A CANARY OF PAIRED DATA
# data_str="changeme"
# data_path=vars_dict["canary_path"]+data_str
# masked=apply_mask_register(vars_dict["canary_path"],"mask_indices_0.p",data_path)


