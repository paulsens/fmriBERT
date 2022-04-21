
proj_dir = "/Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT/"
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

opengenre_samples_per_run = 40

#number of runs for each task in the opengenre dataset
runs_dict = {"Test": 6, "Training": 12}

#path to subject folders containing .tsv event files from opengenre dataset, used by make_opengenre_labels
# i.e, tsv files should be at opengenre_events_path/sub-00X/func/
opengenre_events_path = "/Volumes/External/fmribertfix/opengenredata/bids/"

#path to preprocessed functional bold data of opengenre dataset, the pickled label files are output in each subject's folder here
opengenre_preproc_path = "/Volumes/External/opengenre/preproc/"

COOL_DIVIDEND=420-3 #sweetspot for voxel space dimension, will vary by ROI, but STG is 420 (after the 3 token dimensions are inserted)

ATTENTION_HEADS = 2