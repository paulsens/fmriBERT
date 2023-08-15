import pickle
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
path1 = "/Volumes/External/"
path2 = "/preproc/"
path3 ="/STG_masks/"
ant_f = "STG_ant_linearresamp.nii.gz"
post_f = "STG_post_linearresamp.nii.gz"
sub="001"
dataset="opengenre"
subject = "sub-sid" + sub
fullpath = path1 + dataset + path2 + subject + "/" + subject + path3
# threshold="23"
# left_p = open(fullpath + "STGbinary_left_t" + str(threshold) + ".p", "rb")
#
# left_vol = pickle.load(left_p)
#
# count=0
# for x in range(0, 65):
#     for y in range(0, 77):
#         for z in range(0, 65):
#             if left_vol[x][y][z]==1:
#                 count+=1
#
# print("count is "+str(count))
from Constants import *
hemisphere="left"
threshold="23"
for sub in ["001", "002", "003", "004", "005"]:
    iter = 0  # iterations of the next loop, resets per subject
    # opengenre_preproc_path is defined in Constants.py
    subdir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub + "/"
    # load voxel data and labels
    with open(subdir + "STG_allruns" + hemisphere + "_t" + threshold + ".p", "rb") as data_fp:
        all_data = pickle.load(data_fp)
    with open(subdir + "labelindices_allruns.p", "rb") as label_fp:
        all_labels = pickle.load(label_fp)

def make_pretraining_data(threshold, hemisphere, seq_len=5, num_copies=1, test_copies=4, allowed_genres=range(0,10), standardize=1, detrend="linear", within_subjects=1, binary="same_genre", multiclass="genre_decode", verbose=1):
    #runs_dict is defined in Constants.py
    test_runs = runs_dict["Test"]
    training_runs = runs_dict["Training"]

    #size of left or right STG voxel space, with 3 token dimensions already added in
    voxel_dim = COOL_DIVIDEND+3 #defined in Constants.py
    CLS = [1] + ([0] * (voxel_dim - 1))  # first dimension is reserved for cls_token flag
    MSK = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
    SEP = [0, 0, 1] + ([0] * (voxel_dim - 3))  # third dimension is reserved for sep_token flag

    #each element of this list should be (seq_len*2 + 2,real_voxel_dim+3), i.e CLS+n TRs+SEP+n TRs
    #   where 3 extra dimensions have been added to the front of the voxel space for the tokens
    training_samples = []
    # final list of labels for training
    training_labels = []
    #keep count of how many sample/label pairs we've created
    count = 0

    # the reference list of left-hand samples, index set by the count variable
    #  find its genre with ref_to_genre_dict
    ref_samples = []
    # reference list of genres, where element i is the genre of ref_samples[i]
    ref_genres = []

    #a list of indices for each genre, where to find each genre in the aggregate training data list
    genre_sample_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    ref_to_genre_dict = {}

    #loop through subjects
    for sub in ["001", "002", "003", "004", "005"]:
        iter=0 #iterations of the next loop, resets per subject
        #opengenre_preproc_path is defined in Constants.py
        subdir = opengenre_preproc_path + "sub-sid" + sub + "/" + "sub-sid" + sub + "/"
        #load voxel data and labels
        with open(subdir+"STG_allruns"+hemisphere+"_t"+threshold+".p", "rb") as data_fp:
            all_data = pickle.load(data_fp)
        with open(subdir+"labelindices_allruns.p","rb") as label_fp:
            all_labels = pickle.load(label_fp)
        voxel_data=[]
        labels=[]

        #only remove the test_copies many repetitions (max 4) during the Test Runs
        n_test_runs = runs_dict["Test"]
        amount = OPENGENRE_TRSPERRUN*test_copies//4
        for run in range(0, n_test_runs):
            start = run*OPENGENRE_TRSPERRUN
            for TR in range(start, start+amount):
                voxel_data.append(all_data[TR])
                if(TR%10==0): # each label corresponds to 10 TRs
                    labels.append(all_labels[TR//10]) # index in the labels list is one tenth floor'd of the data index
        #add every TR from the remaining 12 Training Runs, no repetitions here
        start = n_test_runs*OPENGENRE_TRSPERRUN
        for TR in range(start, len(all_data)):
            voxel_data.append(all_data[TR])
            if (TR % 10 == 0):  # each label corresponds to 10 TRs
                labels.append(all_labels[TR // 10])  # index in the labels list is one tenth floor'd of the data index
        #labels are 0 through 9 inclusive
        #print("length of voxel data after applying test_copies "+str(len(voxel_data)))
        #print("length of labels after applying test_copies is "+str(len(labels)))


        #if the standardize flag is set, set mean to zero and variance to 1
        #function is defined in helpers.py
        # if(standardize and within_subjects):
        #     voxel_data = standardize_flattened(voxel_data)

        # timesteps = len(voxel_data)
        # #we're going to obtain timesteps//seq_len many samples from each subject
        # for t in range(0, timesteps//seq_len):
        #     start = t*seq_len
        #     this_sample = [] #create the left-hand sample starting at TR number {start}
        #     # add the seq_len many next images
        #     for s in range(0, seq_len):
        #         this_sample.append(voxel_data[start+s])
        #
        #     this_genre = labels[iter//(10//seq_len)] #the divisor is either 2 or 1
        #
        #     #add count as an index corresponding to this genre
        #     genre_sample_dict[this_genre].append(count)
        #     ref_to_genre_dict[count]=this_genre
        #     #append to aggregate lists
        #     ref_samples.append(this_sample)
        #     ref_genres.append(this_genre)
        #     #now ref_samples[i] is a left hand sample with genre ref_genres[i]
        #
        #     #increase count
        #     count = count+1
        #     iter = iter+1

def detrend_flattened(voxel_data, detrend, sub):
    # each voxel is detrended independently so fix the voxel i.e the column and loop over the timesteps
    n_columns = len(voxel_data[0])
    n_rows = len(voxel_data)
    x_train = range(0, n_rows)
    x_train=np.array(x_train)
    voxel_data = np.array(voxel_data)
    n_plotpoints = 7200
    y_plots=[]
    # dimensions 0, 1, and 2 in voxel space are just token dimensions, don't detrend those
    for voxel in range(3, n_columns):
        y_train = voxel_data[:,voxel]
        y_plot = np.zeros(n_plotpoints)
        if(sub=="001" and voxel==200):
            maxv = max(x_train)
            minv = min(x_train)
            # whole range we want to plot
            x_plot = np.linspace(0, n_rows-1, n_plotpoints)
            for i in range(0, n_plotpoints):
                x_plot[i]=x_plot[i]//1
                y_plot[i] = y_train[int(x_plot[i])]
            # create 2D-array versions of these arrays to feed to transformers
            X_train = x_train[:, np.newaxis]
            X_plot = x_plot[:, np.newaxis]

            # plot function
            lw = 2
            fig, ax = plt.subplots()
            fig2, ax2 = plt.subplots()
            fig3, ax3 = plt.subplots()
            fig4, ax4 = plt.subplots()
            fig5, ax5 = plt.subplots()
            ax.set_prop_cycle(
                color=["black", "teal", "yellowgreen", "gold", "darkorange", "tomato"]
            )
            ax.plot(x_train, y_train, linewidth=lw, label="ground truth")

            # polynomial features
            for degree in [1, 2, 3]:
                model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
                model.fit(X_train, y_train)
                y_plot = model.predict(X_plot)
                y_plots.append(y_plot.copy())
                ax.plot(x_plot, y_plot, label=f"degree {degree}")

            # B-spline with 4 + 3 - 1 = 6 basis functions
            model = make_pipeline(SplineTransformer(n_knots=5, degree=3), Ridge(alpha=1e-3))
            model.fit(X_train, y_train)

            y_plot = model.predict(X_plot)
            y_plots.append(y_plot.copy())
            vmax=max(y_train)
            vmin=min(y_train)
            ax.plot(x_plot, y_plot, label="B-spline")
            ax.legend(loc="lower center")
            ax.set_ylim(vmin-5, vmax+5)

            ax2.plot(x_plot, y_train - y_plots[0], c="teal")
            ax3.plot(x_plot, y_train - y_plots[1], c="yellowgreen")
            ax4.plot(x_plot, y_train - y_plots[2], c="gold")
            ax5.plot(x_plot, y_train - y_plots[3], c="darkorange")
            plt.show()


if __name__=="__main__":
    #seq_len should be either 5 or 10
    #hemisphere is left or right
    #threshold only 23 right now
    #num_copies is the number of positive and negative training samples to create from each left-hand sample
    #allowed_genres is a list of what it sounds like, remember range doesn't include right boundary
    make_pretraining_data("23", hemisphere="left", seq_len=5, num_copies=5, standardize=1, detrend="spline", within_subjects=1, allowed_genres=range(0,10))