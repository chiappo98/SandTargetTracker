#!/bin/bash

export HOME=$(pwd)

SCRIPT_PATH=/storage/gpfs_data/neutrino/users/alrugger/Software/dchamber_prototype_analysis/cluster_analysis
export PYTHONPATH=$HOME/site-packages:$PYTHONPATH

python3 $SCRIPT_PATH/fit_toy_data.py --n_jobs ${1} --job_idx ${2} --input_path ${3} --output_path ${4} --base_name ${5}