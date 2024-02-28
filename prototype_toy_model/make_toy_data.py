# library imports
import numpy as np
import math as mt
import argparse
import time
import pandas as pd

# module import
from common_functions import *

# Toy model functions
def generate_track():
    """Extracts random parameters for the tracks

    Returns: track parameters in a [[r0],theta,phi] format.
    """
    rnd_x = np.random.uniform(4, 20)
    rnd_y = np.random.uniform(0, 30)
    r_0 = [rnd_x, rnd_y, 3]
    # extract random angles
    rnd_theta = -0.5*mt.pi#np.random.uniform(-0.5 * mt.pi, -0.125 * mt.pi)
    rnd_phi = 0.0#np.random.uniform(-0.5 * mt.pi, 1.5 * mt.pi)

    return [r_0, rnd_theta, rnd_phi]

def simulate_signals(track_pars, wire_dict):
    """Simulates the timing signals from the chamber given some track parameters.

    Args:
        track_pars: parameters of the incident track.
    Returns: active-flag and times lists to be added to dataframe entry
    """
    # data are saved as 1D lists, going from layer_0 to layer_2, appending the values each time (possibly)
    flag_list = []
    time_list = []

    # compute the wire spacing and cell height
    wire_spacing = wire_dict["wire_spacing"]
    cell_height = wire_dict["cell_height"]
    # compute the max possible distance from the wire in a cell
    max_dist = np.linalg.norm([0.5 * wire_spacing, 0.5 * cell_height])

    # cycle over the wires and check for intersection in the cell: compute timing if intersected
    # for layer_key,layer in wire_dict.items():
    for wire_key, wire in wire_dict["wires"].items():
        # distance between wire and intersect on the upper strip layer
        strip_z = wire["r0"][2] + 0.5 * cell_height
        upper_strip_intersect = track_plane_intersec(
            track_pars, [[0, 0, strip_z], [0, 0, 1.0]]
        )
        z_inter_dist = point_wire_distance(upper_strip_intersect, wire)

        # z of the intersect between track and cell boundary planes
        plane_norm = np.cross(wire["e"], [0.0, 0.0, 1.0])
        lateral_inter_dist = []
        int_list = []
        for i in [-1, 1]:
            # compute and append the distance of the intersect points from the wire
            intersect = track_plane_intersec(
                track_pars,
                [[wire["r0"] + np.asarray([0.5 * i * wire_spacing, 0, 0])], plane_norm],
            )
            lateral_inter_dist.append(point_wire_distance(intersect, wire))
            int_list.append(intersect)
        # impose constraints on the intersections and fill the lists
        if (
            lateral_inter_dist[0] <= max_dist
            or lateral_inter_dist[1] <= max_dist
            or z_inter_dist <= max_dist
        ):
            # this could be an and to the previous if
            if (
                0 <= int_list[0][1] <= 30.0 or 0 <= int_list[1][1] <= 30
            ) or 0 <= upper_strip_intersect[1] <= 30.0:
                flag_list.append(1.0)
                time_list.append(
                    simulate_time(
                        track_wire_distance(track_pars, wire),
                        len_along_wire(track_pars, wire),
                        True,
                    )
                )
                # print("Wire ",wire_key,": active")
            else:
                flag_list.append(0.0)
                time_list.append(0.0)
        else:
            flag_list.append(0.0)
            time_list.append(0.0)

    return flag_list, time_list

def parse_args():
    parser = argparse.ArgumentParser(description="Track generation selection")
    parser.add_argument(
        "--save_path",
        required=False,
        help="Save path of the dataframe",
        type=str,
        default="../dataframes/raw",
    )
    parser.add_argument(
        "--df_name",
        required=False,
        help="Base string of the dataframe name",
        type=str,
        default="newgeo_redu_th0_phi0",
    )
    parser.add_argument(
        "--n_tracks",
        required=False,
        help="How many tracks to generate",
        type=int,
        default=20,
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("-- Main --")

    print("> Parsing arguments")
    args = parse_args()
    save_path = args.save_path
    base_string = args.df_name
    n_tracks = args.n_tracks

    
    wire_dict = make_config_dictionary()

    np.random.seed(0)
    event_lst = []
    start = time.time()
    print("> Generating tracks")
    for i in range(0, n_tracks):
        entry_dict = {}
        track_pars = generate_track()
        # print("-- Simulating track {0} --".format(i))
        flag_lst, time_lst = simulate_signals(track_pars, wire_dict)
        entry_dict.update(
            {"track_pars": track_pars, "flag_lst": flag_lst, "time_lst": time_lst}
        )
        event_lst.append(entry_dict)
    end = time.time()
    print("Elapsed: ",end-start)
    print("> Saving dataframe")
    # define a dataframe for storing the simulation data[[track_pars],[flag_lst],[times_lst]]
    save_frame = pd.DataFrame(event_lst)
    save_frame.to_pickle("{0}/{1}_nev{2}_data.pkl".format(save_path,base_string,n_tracks))