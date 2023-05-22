import sys, os




def addtodict(fold, dict):
    for epoch in range(0, 5):
        # list containing average validation loss and average validation accuracy
        dict[(fold,epoch)]=[0,0]

def averagefold(fold, dict):
    for epoch in range(0, 5):
        loss_sum = dict[(fold,epoch)][0]
        acc_sum = dict[(fold,epoch)][1]

        # each fold had 5 runs contributing to this sum
        # unless fold is fresh, and then it's 10
        # runs_per_fold is set in main code in this file
        divisor = runs_per_fold

        loss_average = loss_sum/divisor
        acc_average = acc_sum/divisor

        dict[(fold,epoch)][0] = loss_average
        dict[(fold, epoch)][1] = acc_average

def getlossacc(line):
    stuff = line.strip().split(": ")
    acc = float(stuff[-1])
    loss = stuff[-2].split(" |")
    loss=float(loss[0])
    return loss,acc

def updateaverages(fold, epoch, loss, acc, dict):
    dict[(fold, epoch)][0] += loss
    dict[(fold, epoch)][1] += acc

def get_averages(fold_averages, ofile_dir):
    fold = 0
    count = 0
    epoch = 0
    addtodict(fold, fold_averages)
    print(ofile_dir)
    for ofile in os.listdir(ofile_dir):
        if ofile == "info.txt" or ofile[0] == ".":
            continue
        ofile_path = ofile_dir + ofile
        with open(ofile_path, "r") as ofile_fp:

            info = ofile_fp.readlines()
            epoch = 0

            line = info[13]

            loss, acc = getlossacc(line)

            updateaverages(fold, epoch, loss, acc, fold_averages)

            epoch += 1

            line = info[25]
            loss, acc = getlossacc(line)
            updateaverages(fold, epoch, loss, acc, fold_averages)
            epoch += 1

            line = info[37]
            loss, acc = getlossacc(line)
            updateaverages(fold, epoch, loss, acc, fold_averages)
            epoch += 1

            line = info[49]
            loss, acc = getlossacc(line)
            updateaverages(fold, epoch, loss, acc, fold_averages)
            epoch += 1

            line = info[61]
            loss, acc = getlossacc(line)
            updateaverages(fold, epoch, loss, acc, fold_averages)
            epoch += 1

            count += 1

            if count % runs_per_fold == 0:
                averagefold(fold, fold_averages)
                fold += 1
                if(fold < 12):
                    addtodict(fold, fold_averages)


def average_the_folds(fold_averages, task):
    # average the fresh fold

    #### do stuff with all the averages
    folds = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    num_folds = 12
    if(task=="fresh"):
        folds=(0,)
        num_folds=1

    # lists of averages across folds, indexed by epoch
    loss_averages = [0, 0, 0, 0, 0]
    acc_averages = [0, 0, 0, 0, 0]

    for fold in range(0, num_folds):
        for epoch in range(0, 5):
            averages = fold_averages[(fold, epoch)]
            loss_avg = averages[0]
            acc_avg = averages[1]

            loss_averages[epoch] += loss_avg
            acc_averages[epoch] += acc_avg

    for epoch in range(0, 5):
        temploss = loss_averages[epoch]
        tempacc = acc_averages[epoch]
        loss_averages[epoch] = temploss / num_folds
        acc_averages[epoch] = tempacc / num_folds

    return loss_averages,acc_averages



###### MAIN CODE
bothfold_averages={}
freshfold_averages = {}

tasks = ["both", "fresh"]
ofile_dir_base = "/Volumes/External/opengenre/samegenre/nov17/"
dict = None

for task in tasks:
    ofile_dir = ofile_dir_base+task+"/"
    if task=="both":
        dict=bothfold_averages
        runs_per_fold=5
    elif task=="fresh":
        dict=freshfold_averages
        runs_per_fold=5

    get_averages(dict, ofile_dir)

print(bothfold_averages)
print(freshfold_averages)
for task in tasks:
    dict=None
    if task=="both":
        dict=bothfold_averages
    elif task=="fresh":
        dict=freshfold_averages
    loss_avgs, acc_avgs = average_the_folds(dict, task)

    print("For task "+str(task)+":")
    print("Loss averages by epoch: "+str(loss_avgs))
    print("Acc averages by epoch: "+str(acc_avgs))


# print stuff with folds separated so i can calculate cross-validation
both=bothfold_averages
fresh=freshfold_averages

# For each fold, for each epoch, average over the 5 iterations
for fold in range(0,12):
    print("FOR FOLD {0}:".format(fold))
    for epoch in range(0,5):
        print(both[fold,epoch],end="\t")
    print("\n")
    for epoch in range(0,5):
        print(fresh[fold,epoch],end="\t")

    print("\n")




