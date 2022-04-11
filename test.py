# import pickle
# import numpy as np
# from Constants import *
# #from transformer import *
# from random import randint
#
# import os, sys
#
# #rootdir = "/Volumes/External/pitchclass/preproc/"
# rootdir = "/Volumes/External/brainlife/opengenre/"
subjects = ["1088", "1125", "1401", "1410", "1419", "1427", "1541", "1571", "1581", "1660", "1661", "1664", "1665",
            "1668", "1672", "1678", "1680"]

# id="1"
# #fullid = "sub-sid00" + id
# fullid = "sub-00"+id
# path = rootdir+fullid+"/"+fullid+"/"
# dirlist = (os.listdir(path))
# for dir in dirlist:
#     newdir = dir[:-25]
#     newdir = path+newdir
#     olddir = path+dir
#     print(str(olddir)+" becomes "+str(newdir))
#     os.rename(olddir, newdir)

myarray = [ [1, 2, 3], [4, 5, 6] ]

print(myarray[0][1])