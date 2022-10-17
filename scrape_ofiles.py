import sys, os

ofile_dir = "/Volumes/External/opengenre/official/ofiles/"
tasks = ["both","binaryonly","multionly"]
val_lines_only = [19, 34, 49, 64, 79]
correct_lines_only = [22, 37, 52, 67, 82]

val_lines_both = [20, 37, 54, 71, 88]
loss_lines_both = [21, 38, 55, 72, 89]
correct_lines_both = [24, 41, 58, 75, 93]

val_lines_d = {
    "both":val_lines_both,
    "binaryonly":val_lines_only,
    "multionly":val_lines_only,
}

def scrape_both(all_lines):

def scrape_binary(all_lines):

def scrape_multi(all_lines):

if __name__ == "__main__":
    for task in tasks:
        task_path = ofile_dir+task+"/"
        lines_list = val_lines_d[task]

        for filename in os.listdir(task_path):
            if(filename[0]!="."):
                ofile_path = task_path+filename
                with open(ofile_path, "r") as ofile_fp:
                    all_lines = ofile_fp.readlines()
                    index = all_lines[2]

                    if task=="both":
                        scrape_both(all_lines)
                    elif task=="binaryonly":
                        scrape_binary(all_lines)
                    elif task=="multionly":
                        scrape_multi(all_lines)



