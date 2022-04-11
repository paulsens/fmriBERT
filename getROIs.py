import nibabel as nib
import itertools

rootdir = "/Volumes/External/pitchclass/preproc/"
subjects = ["1088","1125","1401","1410","1419","1427","1541","1571","1581","1660","1661","1664","1665","1668","1672","1678","1680"]
path = rootdir+"sub-sid001088/sub-sid001088/funcmasks/"
imgpath=path+"func_STG_mask.nii.gz"
fullpath = "/Volumes/External/fmribertfix/opengenredata/bids/sub-001/func/sub-001_task-Test_run-01_bold.nii"
#fullpath = "/Volumes/External/casey_pitchclass/discovery_all/sub-sid001401/func/sub-sid001401_task-pitchheard_run-01_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz"
img = nib.load(fullpath)

data = img.get_fdata()
xsize = len(data)
ysize = len(data[0])
zsize = len(data[0][0])
count=0

# for x in range(0, xsize):
#     for y in range(0, ysize):
#         for z in range(0, zsize):
#             value = data[x][y][z]
#             if value>=5:
#
#                 count+=1

print(count)
print(data.shape)
