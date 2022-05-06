import torch
import torch.nn as nn
from random import randint
import numpy as np
from helpers import *
from voxel_transformer import *
from pitchclass_data import *
from torch.utils.data import DataLoader
import torch.optim as optim
from Constants import *
import sys
import os
import datetime

if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        thiscount=None #gets changed if a count is passed as a command line argument
        #get command line arguments and options
        opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
        args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
        if "-m" in opts:
            # -m "this is the description of the run" will be at the end of the command line call
            run_desc = args[-1]
        else:
            run_desc = None
        if "-gpu" in opts:
            gpunum=args[-2] #currently only works if only one gpu is given
            device = torch.device("cuda:"+str(gpunum))
        else:
            device="cpu"
        if "-count" in opts:
            thiscount=int(args[-2])
        today = datetime.date.today()
        now = datetime.datetime.now()
        ##############################  SET PARAMETERS  ##############################
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #dictionary of hyperparameters, eventually should probably come from command line
        hp_dict={
            "COOL_DIVIDEND" : COOL_DIVIDEND,
            "ATTENTION_HEADS" : ATTENTION_HEADS,
            "device" : str(device),
            "MSK_flag" : 1,
            "CLS_flag" : 1,
            "BATCH_SIZE" : 1,
            "EPOCHS" : EPOCHS,
            "LEARNING_RATE" : 0.00005,
            #Have to manually set the name of the folder whose training data you want to use, since there will be many
            "data_dir" : "2022-05-03",
            #Manually set the hemisphere and iteration number of the dataset you want to use in that folder
            "hemisphere": "left",
            "count" : "0",
            #manually set max_seq_length used in data creation, in the input CLS+seq+SEP+seq this is the max length of seq
            "max_sample_length":5
        }
        hp_dict["data_path"] = opengenre_preproc_path + "training_data/" + hp_dict["data_dir"] + "/"
        torch.set_default_dtype(torch.float64)

        #set up logfile, PRETRAIN_LOG_PATH is defined in Constants.py
        today_dir = PRETRAIN_LOG_PATH+str(today)+"/"
        if not (os.path.exists(today_dir)):
            os.mkdir(today_dir)
        if(thiscount!=None):
            logcount=thiscount
        else:
            logcount=0
        logfile = today_dir + "pretrain_log_"+str(logcount)+".txt"
        while (os.path.exists(logfile)):
            logcount+=1
            logfile = today_dir + "pretrain_log_" + str(logcount) + ".txt"
        log = open(logfile,"w")
        log.write(str(now)+"\n")
        # run_desc is potentially given in command line call
        if(run_desc is not None):
            log.write(run_desc+"\n\n")
            print(run_desc+"\n\n")
        #write hyperparameters to log
        for hp in hp_dict.keys():
            log.write(str(hp)+" : "+str(hp_dict[hp])+"\n")

        #load samples and labels
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_samples"+hp_dict["count"]+".p", "rb") as samples_fp:
            train_X = pickle.load(samples_fp)
        with open(hp_dict["data_path"] + hp_dict["hemisphere"] + "_labels"+hp_dict["count"]+".p", "rb") as labels_fp:
            train_Y = pickle.load(labels_fp)

    rows = len(train_X)
    #columns = len(train_Y[0])
    print(train_Y)