import pandas as pd

bidsdir = "/Volumes/External/fmribertfix/opengenredata/bids/"

fullpath = bidsdir + "sub-004/func/sub-004_task-Test_run-01_events.tsv"

events_df = pd.read_csv(fullpath, sep='\t')

print(events_df["poop"])

