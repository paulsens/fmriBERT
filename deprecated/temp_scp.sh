#!/bin/bash

cd /Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT/

scp pairedpretrain.script pairedpretrain.sh f003543@discovery7.hpcc.dartmouth.edu:/isi/music/auditoryimagery2/seanthesis/thesis/scripts
scp pairedfinetune.script pairedfinetune.sh f003543@discovery7.hpcc.dartmouth.edu:/isi/music/auditoryimagery2/seanthesis/thesis/scripts
scp unpaireddecode.script unpaireddecode.sh f003543@discovery7.hpcc.dartmouth.edu:/isi/music/auditoryimagery2/seanthesis/thesis/scripts
scp unpairedpretrain.script unpairedpretrain.sh f003543@discovery7.hpcc.dartmouth.edu:/isi/music/auditoryimagery2/seanthesis/thesis/scripts

