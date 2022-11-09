import pickle
import random

data_path = "/Volumes/External/pitchclass/preproc/"
debug=1
hemisphere="left"
threshold="23"
ROI="STG"
n_runs=8 #all subjects had 8 runs in this dataset
sublist = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665", "1668", "1672", "1678", "1680"]
ROIfile_str=ROI+"_allruns"+str(hemisphere)+"_t"+str(threshold)+".p"

subid = "sub-sid001088"
full_path = data_path + subid + "/" + subid + "/" + ROIfile_str
save_path = "/Volumes/External/pitchclass/finetuning/sametimbre/datasets/2/"
opengenre_path = "/Volumes/External/opengenre/preproc/training_data/2022-06-12/left_samples3.p"

all_save_path = save_path + "all_"
# with open(all_save_path + "X.p", "rb") as temp_fp:
#     all_X = pickle.load(temp_fp)

with open(opengenre_path, "rb") as temp_fp:
    all_X = pickle.load(temp_fp)

print(all_X[1348])
print(all_X[1194])

# potentials = []
# for i in range(0, len(all_X)):
#     potentials.append(i)
# choices = random.sample(potentials, 10)
# print(len(all_X))
# for choice in choices:
#     print("choice is "+str(choice))
#     print(all_X[choice])