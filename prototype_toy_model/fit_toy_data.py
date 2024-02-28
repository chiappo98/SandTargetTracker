# library imports
import argparse
import os
import numpy as np
import math as mt
import pandas as pd
from iminuit import Minuit

# module import
from common_functions import *

def get_chunk_idx(df_len,n_jobs,job_idx):
    """Extracts the dataframe indices correspoding to the current job

    Args:
        df_len: length of the dataframe.
        n_jobs: total number of jobs.
        job_idx: job index within the queue.

    Returns: indices to be fitted by the job.
    """
    # sanity check
    assert df_len > n_jobs and n_jobs > job_idx, "Job idx. overflow: check number of jobs and index!"
    
    # split the list of rows into n_jobs and take the job_idx-th element
    idx_array = np.arange(df_len)
    return np.array_split(idx_array,n_jobs)[job_idx]

def get_cross(a_wire, b_wire):
    """Computes the crossing points between two cells

    Args: 
        a_wire, b_wire: parameters of the two signal wires.
    Returns:
        coord_lst: list of the cross coordinates.
    """
    coord_lst = []
    for i in [-1.0, 1.0]:
        a = a_wire + np.array([0, 0, 0, i, 0, 0])
        for j in [-1.0, 1.0]:
            b = b_wire + np.array([0, 0, 0, j, 0, 0])
            if np.any(np.dot(b[:3], a[:3]) != 1):
                w_len = (
                    np.dot((a[3:] - b[3:]), (a[:3] - b[:3] * np.dot(b[:3], a[:3])))
                ) / ((np.dot(b[:3], a[:3])) ** 2 - 1)
            else:
                w_len = 0
            coord_lst.append(w_len * a[:3] + a[3:])
    return coord_lst

def get_limits(wire_dict):
    """Computes the fit constraints on (x,y) from the wire intersections.

    Args:
        wire_dict: dictionary of the active cells.
    Returns:
        x_lims, y_lims: intersection coordinate arrays.
    """
    # extract an array of parameters from the dictionary
    wire_params = np.empty((0))
    for wire_key, wire in wire_dict.items():
        wire_params = np.append(wire_params,np.concatenate((wire["pars"]["e"],wire["pars"]["r0"])),axis=0)
    wire_params=wire_params.reshape((-1,6))

    # compute the crossing points if 2 or more cells are active
    coor_lst = []
    if wire_params.shape[0] == 1:
        coor_lst.append([[wire_params[:,3:] - [1.0, 0, 0]], [wire_params[:,3:] + [1.0, 0, 0]]])
    else:
        for i in range(wire_params.shape[0]):
            for j in range(i + 1, wire_params.shape[0]):
                coor_lst.append(get_cross(wire_params[i], wire_params[j]))
    coor_lst = np.reshape(np.asarray(coor_lst), (-1, 3))

    # then take the min and max in the subset
    cross_sort = coor_lst[coor_lst[:, 0].argsort()]
    x_lims = [
        round(max(4.0, cross_sort[0, 0]), 6),
        round(min(20.0, cross_sort[-1, 0]), 6),
    ]
    y_lims = [
        round(max(0.0, np.min(cross_sort[:, 1])), 6),
        round(min(30.0, np.max(cross_sort[:, 1])), 6),
    ]
    return np.asarray(x_lims), np.asarray(y_lims)

def chisq_function(t_p, wire_dict) -> float:
    """Computes the Chi-squared cost function on the 
    """
    track_pars = [[t_p[0], t_p[1], 3.0], -0.5 * mt.pi, 0.0]
    chisq = 0
    # modified the chisq function to sum relative residuals
    for wire_key, wire in wire_dict.items():
        chisq += (
            (wire["m_time"]
            - simulate_time(
                track_wire_distance(track_pars, wire["pars"]),
                len_along_wire(track_pars, wire["pars"]),
                False,
            ))/16
        ) ** 2
    return chisq

def fit_dataframe_tracks(df_entry, wire_dict):
    """Fits the dataframe time signals to tracks
        
    Args: 
        df_entry: row of the dataframe containing data of a single event.
        wire_dict: wire configuration dictionary.
        
    Returns: the fitted track parameters.
    """
    # get the active channel indices for the event
    active_channels = np.argwhere(np.asarray(df_entry["flag_lst"]) > 0)[:, 0]
    # slice the times, and parameter arrays with the active channel indices
    times_array = np.asarray(np.asarray(df_entry["time_lst"])[active_channels])
    # extract the dictionary of active wires
    active_dict= {key: {"pars":wire_dict["wires"][key],"m_time": np.asarray(df_entry["time_lst"])[key]} for key in active_channels}

    # determine the intersections between the wires
    x_lims, y_lims = get_limits(active_dict)
    # vector of fit parameters first guesses
    fit_guess = [x_lims.sum() / 2, y_lims.sum() / 2]
    # #fit_guess = [round(df_entry['track_pars'][0][0],4),round(df_entry['track_pars'][0][1],4)]

    def chi_lambda(pars): return chisq_function(
        pars, active_dict)

    min = Minuit(chi_lambda, fit_guess)
    min.errordef = Minuit.LEAST_SQUARES
    min.limits = [(x_lims[0], x_lims[1]), (y_lims[0], y_lims[1])]
    # standard ncall = 250
    #min.scan().migrad()
    min.scan(ncall=4000).scipy()
    fitted_track_pars = [
        [min.values[0], min.values[1], 3.0],
        -0.5 * mt.pi,
        0.0,
    ]

    return fitted_track_pars

def parse_args():
    parser = argparse.ArgumentParser(description="Dataframe fitting selection")
    parser.add_argument(
        "--n_jobs",
        required=True,
        help="Number of overall jobs",
        type=int,
    )
    parser.add_argument(
        "--job_idx",
        required=True,
        help="Index of the present job",
        type=int,
    )
    parser.add_argument(
        "--input_path",
        required=False,
        help="Path of the input dataframe",
        type=str,
        default="../dataframes/raw/newgeo_redu_th0_phi0_nev90_data.pkl",
    )
    parser.add_argument(
        "--output_path",
        required=False,
        help="Path of the output dataframe chunks",
        type=str,
        default="../dataframes/fitted",
    )
    parser.add_argument(
        "--base_name",
        required=False,
        help="Common name string for the chunks.",
        type=str,
        default="nosmr_redu_th0_phi0",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    print("-- Df fitting Main --")
    # parse the arguments
    args = parse_args()
    n_jobs = args.n_jobs
    job_idx = args.job_idx
    in_path = args.input_path
    out_path = args.output_path
    base_name = args.base_name

    # create the output folder if t doesn't exist already
    if not os.path.exists(out_path):
        print("> Creating new output folder")
        os.makedirs(out_path)
    else:
        print("> Output folder already exists")

    # import the wire configuration dictionary
    wire_dict = make_config_dictionary()

    print("> Reading chunk {0} of df:{1}".format(job_idx,in_path))
    # import the dataframe and extact chunk corresponding to the job
    evt_df = pd.read_pickle(in_path)
    n_evts = len(evt_df)
    idxs = get_chunk_idx(n_evts,n_jobs,job_idx)
    evt_df = evt_df.loc[idxs]
    
    # define the fit lambda function
    def lambda_fit(df_entry): return fit_dataframe_tracks(df_entry, wire_dict)
    # fit the events
    print("> Fitting input df chunk")
    evt_df["fit_pars"] = evt_df.apply(lambda_fit, axis=1)
    # save the resulting dataframe
    df_path = "{0}/{1}_nev{2}_nj{3}_jidx{4}.pkl".format(out_path,base_name,n_evts,n_jobs,job_idx)
    print("> Saving df chunk to: ",df_path)
    evt_df.to_pickle(df_path)
    


    


