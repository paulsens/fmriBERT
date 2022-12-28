from pretrain_results import *


BOTH_dict = {}

NTP_dict = {}

MBM_dict = {}

for i in range(0, len(BOTH_keys)):
    key = BOTH_keys[i]
    BOTH_dict[key] = i
for i in range(0, len(NTP_keys)):
    key = NTP_keys[i]
    NTP_dict[key] = i
for i in range(0, len(MBM_keys)):
    key = MBM_keys[i]
    MBM_dict[key] = i


def get_average(folds_dict, task_dict, key, epoch_key, task):
    avg1 = 0
    avg2 = None
    count = 0
    index = task_dict[key]

    folds = folds_dict[epoch_key]
    for fold in folds:
        fold_info = fold.split(" ")
        info = fold_info[index]
        if key == "train_correct" or key == "val_correct":
            if avg2 is None:
                avg2 = 0
            neg_corr = info.split(",")[0]
            pos_corr = info.split(",")[1]
            avg1 += float(neg_corr)
            avg2 += float(pos_corr)

        else:
            value = float(info)
            avg1 += value

        count +=1
    avg1 = avg1/count
    print("Average for "+task+" "+str(key)+" on "+str(epoch_key))
    print(avg1)
    if avg2 is not None:
        avg2 = avg2/count
        print(avg2)

def get_averages(task, key_list, epoch_key="first_epoch"):
    averages = {}
    for key in key_list:
        if "correct" in key:
            averages[key]=[0,0]
        else:
            averages[key]=0

    for key in averages.keys():
        count = 0
        result_list = task[epoch_key]
        index = list(averages.keys()).index(key)

        for cycle in range(0,len(result_list)):
            cycle_info = result_list[cycle]
            info_list = cycle_info.split(" ")
            value = info_list[index]

            if "correct" in key:
                both = value.split(",")
                averages[key][0]+=float(both[0])
                averages[key][1]+=float(both[1])

            else:
                averages[key]+=float(value)
            count+=1

        if "correct" in key:
            averages[key][0]=averages[key][0]/count
            averages[key][1]=averages[key][1]/count
        else:
            averages[key]=averages[key]/count
    return averages

def sort_cycles(task, key_list, sort_key, epoch_key="fifth_epoch"):
    result_list = task[epoch_key]
    sort_idx =  key_list.index(sort_key)
    pairs = []
    for cycle in range(0, len(result_list)):
        info_list = result_list[cycle].split(" ")
        value = float(info_list[sort_idx])
        this_tuple=(value,cycle)
        pairs.append(this_tuple)

    sorted_pairs = sorted(pairs)
    return sorted_pairs

epoch_key = "first_epoch"
# get_average(BOTH, BOTH_dict, "tb_acc", epoch_key, "BOTH")
# get_average(NTPe5, NTP_dict, "tb_acc", epoch_key, "NTP")
# get_average(BOTH, BOTH_dict, "vb_acc", epoch_key, "BOTH")
# get_average(NTPe5, NTP_dict, "vb_acc", epoch_key, "NTP")
# print(get_averages(BOTH, BOTH_keys))
# print(get_averages(NTPe4, NTP_keys))
# print(get_averages(NTPe5, NTP_keys))
# print(get_averages(MBM, MBM_keys))
task = BOTH
keys = BOTH_keys
sort_key = "vb_acc"
print(sort_cycles(task, keys, sort_key, epoch_key=epoch_key))