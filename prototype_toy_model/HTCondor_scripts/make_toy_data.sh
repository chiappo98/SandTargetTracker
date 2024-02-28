#!/bin/bash

SCRIPT_PATH=/storage/gpfs_data/neutrino/users/alrugger/Software/dchamber_prototype_analysis/cluster_analysis

python3 $SCRIPT_PATH/make_toy_data.py --n_tracks "$1" --save_path "$2" --df_name "$3"