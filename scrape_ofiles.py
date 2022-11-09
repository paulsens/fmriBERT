import sys, os

verbose = False

ofile_dir = "/Volumes/External/opengenre/official/ofiles/"
tasks = ["both","binaryonly","multionly"]
# find the info we want in the two ___only tasks
val_lines_only = [19, 34, 49, 64, 79]
correct_lines_only = [22, 37, 52, 67, 82]

# find info we want when training on both tasks simultaneously
val_lines_both = [20, 37, 54, 71, 88]
# loss lines both is the line with the individual losses for bin and multi, the above only contains the weighted sum loss that is backpropagated
loss_lines_both = [21, 38, 55, 72, 89]
correct_lines_both = [24, 41, 58, 75, 92]

val_lines_d = {
    "both":val_lines_both,
    "binaryonly":val_lines_only,
    "multionly":val_lines_only,
}

def scrape_both():

    task = "both"
    task_path = ofile_dir + task + "/"
    lines_list = val_lines_d[task]

    # index n is a list of the validation accuracy after epoch n for each of the 12 folds
    epoch_accs_ntp=[[],[],[],[],[]]
    epoch_accs_mbm=[[],[],[],[],[]]
    epoch_loss_both=[[],[],[],[],[]]
    epoch_loss_ntp=[[],[],[],[],[]]
    epoch_loss_mbm=[[],[],[],[],[]]
    epoch_correctness=[[],[],[],[],[]]

    for filename in os.listdir(task_path):
        if (filename[0] != "."):
            ofile_path = task_path + filename
            with open(ofile_path, "r") as ofile_fp:
                all_lines = ofile_fp.readlines()
                index = all_lines[2]

            for epoch in range(0, len(val_lines_both)):

                # get the info for doing both tasks at once
                line_num = val_lines_both[epoch]
                # looks like Validation: | Loss: 0.03226 | Acc: 68.500 | Acc2: 1.000
                line = all_lines[line_num]
                # looks like ['Validation: | Loss: 0.03226 | ', '68.500 | Acc2: 1.000']
                line = line.strip().split("Acc: ")

                # messy extraction but whatever
                both_loss = line[0].split("Loss: ")
                both_loss = both_loss[1].split(" |")[0]
                ntp_acc = (line[1].split(" |"))[0]
                mbm_acc = line[1].split("Acc2: ")[1].strip()

                # get the two individual losses
                loss_line_num = loss_lines_both[epoch]
                # looks like: Bin val loss was tensor(0.4831) and multi val loss was tensor(0.0025)
                loss_line = all_lines[loss_line_num]
                # looks like: ['Bin val loss was tenso', '0.4831) and multi val loss was tenso', '0.0025)\n']
                loss_line = loss_line.split("(")
                ntp_loss = loss_line[1].split(")")[0]
                mbm_loss = loss_line[2].split(")")[0]

                # get info on correctness
                correct_line_num = correct_lines_both[epoch]
                # looks like: correct counts for this epoch: [245, 283]
                correct_line = all_lines[correct_line_num]
                # looks like: ['correct counts for this epoch: ', '232, 316]\n']
                correct_line = correct_line.split("[")
                # looks like: ['232', '316]\n']
                correct_line = correct_line[1].split(", ")
                neg_correct = correct_line[0]
                pos_correct = correct_line[1].split("]")[0]
                correctness = (int(neg_correct), int(pos_correct))



                # add info to lists
                epoch_accs_ntp[epoch].append(float(ntp_acc))
                epoch_accs_mbm[epoch].append(float(mbm_acc))
                epoch_loss_both[epoch].append(float(both_loss))
                epoch_loss_ntp[epoch].append(float(ntp_loss))
                epoch_loss_mbm[epoch].append(float(mbm_loss))
                epoch_correctness[epoch].append(correctness)

    if verbose:
        for fold in range(0, len(epoch_accs_ntp[0])):
            print("Fold "+str(fold)+":")
            for epoch in range(0, len(epoch_accs_ntp)):
                print("   Epoch "+str(epoch)+":")
                print("       NTP Accuracy: "+str(epoch_accs_ntp[epoch][fold]))
                print("       NTP Loss: "+str(epoch_loss_ntp[epoch][fold]))
                print("       MBM Accuracy: "+str(epoch_accs_mbm[epoch][fold]))
                print("       MBM Loss: "+str(epoch_loss_mbm[epoch][fold]))
                print("       Both Loss: "+str(epoch_loss_both[epoch][fold]))
                print("       Correctness: "+str(epoch_correctness[epoch][fold]))


    return epoch_accs_ntp, epoch_loss_ntp, epoch_accs_mbm, epoch_loss_mbm, epoch_loss_both, epoch_correctness

def scrape_NTP():
    task = "binaryonly"
    task_path = ofile_dir + task + "/"
    lines_list = val_lines_d[task]

    # index n is a list of the validation accuracy after epoch n for each of the 12 folds
    epoch_accs = [[], [], [], [], []]
    epoch_loss = [[], [], [], [], []]
    epoch_correctness = [[],[],[],[],[]]
    for filename in os.listdir(task_path):
        if (filename[0] != "."):
            ofile_path = task_path + filename
            with open(ofile_path, "r") as ofile_fp:
                all_lines = ofile_fp.readlines()
                index = all_lines[2]

            for epoch in range(0, len(val_lines_only)):
                line_num = val_lines_only[epoch]
                # looks like Validation: | Loss: 0.58668 | Acc: 72.250 | Acc2: 0.000
                line = all_lines[line_num]
                # looks like ['Validation: | Loss: 0.58668 | ', '72.250 | Acc2: 0.000']
                line = line.strip().split("Acc: ")

                # messy extraction but whatever
                ntp_loss = line[0].split("Loss: ")
                ntp_loss = ntp_loss[1].split(" |")[0]
                ntp_acc = (line[1].split(" |"))[0]

                # get info on correctness
                correct_line_num = correct_lines_only[epoch]
                # looks like: correct counts for this epoch: [245, 283]
                correct_line = all_lines[correct_line_num]
                # looks like: ['correct counts for this epoch: ', '232, 316]\n']
                correct_line = correct_line.split("[")
                # looks like: ['232', '316]\n']
                correct_line = correct_line[1].split(", ")
                neg_correct = correct_line[0]
                pos_correct = correct_line[1].split("]")[0]
                correctness = (int(neg_correct), int(pos_correct))

                # add info to lists
                epoch_accs[epoch].append(ntp_acc)
                epoch_loss[epoch].append(ntp_loss)
                epoch_correctness[epoch].append(correctness)



    if verbose:
        for fold in range(0, len(epoch_accs[0])):
            print("Fold " + str(fold) + ":")
            for epoch in range(0, len(epoch_accs)):
                print("   Epoch " + str(epoch) + ":")
                print("       NTP Accuracy: " + str(epoch_accs[epoch][fold]))
                print("       NTP Loss: " + str(epoch_loss[epoch][fold]))
                print("       Correctness: "+str(epoch_correctness[epoch][fold]))

    return epoch_accs, epoch_loss, epoch_correctness

def scrape_MBM():
    task = "multionly"
    task_path = ofile_dir + task + "/"
    lines_list = val_lines_d[task]

    # index n is a list of the validation accuracy after epoch n for each of the 12 folds
    epoch_accs = [[], [], [], [], []]
    epoch_loss = [[], [], [], [], []]
    for filename in os.listdir(task_path):
        if (filename[0] != "."):
            ofile_path = task_path + filename
            with open(ofile_path, "r") as ofile_fp:
                all_lines = ofile_fp.readlines()
                index = all_lines[2]

            for epoch in range(0, len(val_lines_only)):
                line_num = val_lines_only[epoch]
                # looks like Validation: | Loss: 0.00130 | Acc: 1.000 | Acc2: 0.000
                line = all_lines[line_num]
                # looks like ['Validation: | Loss: 0.00130 | ', '1.000 | Acc2: 0.000']
                line = line.strip().split("Acc: ")

                # messy extraction but whatever
                mbm_loss = line[0].split("Loss: ")
                mbm_loss = mbm_loss[1].split(" |")[0]
                mbm_acc = (line[1].split(" |"))[0]

                # add info to lists
                epoch_accs[epoch].append(mbm_acc)
                epoch_loss[epoch].append(mbm_loss)

    if verbose:
        for fold in range(0, len(epoch_accs[0])):
            print("Fold " + str(fold) + ":")
            for epoch in range(0, len(epoch_accs)):
                print("   Epoch " + str(epoch) + ":")
                print("       MBM Cosine Similarity: " + str(epoch_accs[epoch][fold]))
                print("       MBM Loss: " + str(epoch_loss[epoch][fold]))

    return epoch_accs, epoch_loss

if __name__ == "__main__":

    # keep in mind these are for the validation split, not training split
    # from 12fold crossvalidation on both tasks at the same time
    accs_both_NTP, losses_both_NTP, accs_both_MBM, losses_both_MBM, losses_both, correctness_both = scrape_both()

    # from 12fold crossvalidation on Next Thought Prediction only
    accs_NTP, losses_NTP, correctness_NTP = scrape_NTP()

    # from 12fold crossvalidation on Masked Brain Modeling only
    accs_MBM, losses_MBM = scrape_MBM()

    # remember epochs start at 0
    epoch = 4
    folds = []
    for fold in range(0,12):
        temp = (accs_both_NTP[epoch][fold], fold)
        folds.append(temp)
    folds.sort()
    print(folds)

    folds = []
    for fold in range(0,12):
        temp = (accs_NTP[epoch][fold], fold)
        folds.append(temp)
    folds.sort()
    print(folds)

    folds = []
    for fold in range(0,12):
        temp = (losses_MBM[epoch][fold], fold)
        folds.append(temp)
    folds.sort()
    print(folds)

