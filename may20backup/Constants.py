env = "local"
if env == "local":
    proj_dir = "/Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT/"
elif env =="discovery":
    proj_dir = "/isi/music/auditoryimagery2/seanthesis/"
pitch_dir = proj_dir + "pitchclass/"
curr_sub = "sid001088"
stg_file = pitch_dir + curr_sub+"/1030/STGsamples.p"
threshold = 23 #probability threshold of voxel being in ROI

# Music genres with arbitrary indices, this is the first order the genres are listed in Nakai et al.
genre_dict = {
    'blues':0, 'classical':1, 'country':2, 'disco':3, 'hiphop':4,
    'jazz':5, 'metal':6, 'pop':7, 'reggae':8, 'rock':9
}

# Reverse of the above dictionary
genrelabel_dict = {
    v: k for k, v in genre_dict.items()
}


#number of runs for each task in the opengenre dataset
runs_dict = {"Test": 6, "Training": 12}

#path to subject folders containing .tsv event files from opengenre dataset, used by make_opengenre_labels
# i.e, tsv files should be at opengenre_events_path/sub-00X/func/
if env == "local":
    opengenre_events_path = "/Volumes/External/fmribertfix/opengenredata/bids/"
elif env == "discovery":
    opengenre_events_path = "/isi/music/auditoryimagery2/seanthesis/opengenredata/bids/"

#path to preprocessed functional bold data of opengenre dataset, the pickled label files are output in each subject's folder here
if env == "local":
    opengenre_preproc_path = "/Volumes/External/opengenre/preproc/"
elif env == "discovery":
    opengenre_preproc_path = "/isi/music/auditoryimagery2/seanthesis/opengenre/preproc/"

if env == "local":
    PRETRAIN_LOG_PATH = "/Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT/pretrain_logs/"
elif env == "discovery":
    PRETRAIN_LOG_PATH = "/isi/music/auditoryimagery2/seanthesis/pretrain_logs/"

COOL_DIVIDEND=420-3 #sweetspot for voxel space dimension, will vary by ROI, but STG is 420 (after the 3 token dimensions are inserted)

val_split = 0.1

# 40 clips of 15 seconds with 1.5TR means 400 TRs per run.
OPENGENRE_TRSPERRUN = 400

if env == "local":
    ATTENTION_HEADS = 2
    EPOCHS = 2
elif env == "discovery":
    ATTENTION_HEADS = 5
    EPOCHS = 10

#when the binary classi task is samegenre, and there are N samples from each genre,
# then we could potentially have (N-1) yes pairs and another N-1 no pairs
#  the amount we actually take is ((N-1)*SAMEGENRE_CROSSPAIRS_PERCENT)//100
SAMEGENRE_CROSSPAIRS_PERCENT = 100
