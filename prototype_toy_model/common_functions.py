# define a constant for the effective drift velocity [cm/ns]
__v_drift__ = 5e-3
# define a constant for the signal propagation velocity [cm/ns]
__v_signal__ = 20

# library imports
import numpy as np
import json
import math as mt


# fill the array of wire unit vectors and initial pos.
def make_config_dictionary(json_path="/storage/gpfs_data/neutrino/users/alrugger/Software/dchamber_prototype_analysis/geo_config.json"):
    """Reads the chamber configuration JSON and produce the dictionary of wire geometric parameters.

    Args:
        json_path: path of the configuration JSON.

    Returns: `True` if production has succeeded, `False` otherwise.
    """
    # open the JSON file
    json_f = open(json_path)
    # read the JSON file as a dictionary
    geo_config = json.load(json_f)
    json_f.close()
    # print("Read the dictionary:\n",geo_config['layers'].keys())

    # usare count per creare un dizionario di un solo livello!!
    parameter_dict = {"wire_spacing": 2.0, "cell_height": 1.0, "wires": {}}
    n_channel = 0
    for layer_key, layer in geo_config["layers"].items():
        # int_dict = {}
        for nwire in range(layer["n_cells"]):
            parameter_dict["wires"][n_channel] = {
                "r0": [
                    nwire * geo_config["wire_spacing"] + layer["x_off"],
                    0.0,
                    layer["y_off"],
                ],
                "e": [
                    mt.sin(mt.pi * (layer["stereo_angle"] / 180.0)),
                    mt.cos(mt.pi * (layer["stereo_angle"] / 180.0)),
                    0.0,
                ],
            }
            n_channel += 1

    # with open ('./parameter_dict.json','w') as f:
    #     json.dump(parameter_dict,f)
    for item in parameter_dict.items():
        print(item)
    return parameter_dict


def track_wire_distance(track_pars: list, wire_pars: dict) -> float:
    """Computes the distance between a track and a wire given their parameters.

    Args:
        track_pars: list of track parameters [[p0],theta,phi]. Angles in radians.
        wire_pars: wire parameter dict. entry for the wire.

    Returns: the straight-line distance `tw_dist`.
    """
    # compute the track unit-vector
    e_track = np.array(
        [
            mt.cos(track_pars[1]) * mt.cos(track_pars[2]),
            mt.cos(track_pars[1]) * mt.sin(track_pars[2]),
            mt.sin(track_pars[1]),
        ]
    )
    # get the track coordinate-vector
    p_track = np.asarray(track_pars[0])
    # get the wire unit and coordinate vectors
    e_wire = np.asarray(wire_pars["e"])
    r_wire = np.asarray(wire_pars["r0"])
    # compute the wire-track distance
    n = np.cross(e_wire, e_track)
    tw_dist = abs(np.dot(n, (r_wire - p_track))) / np.linalg.norm(n)

    return tw_dist

def len_along_wire(track_pars: list, wire_pars) -> float:
    """
    Computes the point along a wire with the min. distance from a track.

    Arguments
    --------
        track_pars: list of track parameters [[p0],theta,phi]. Angles in radians.
        wire_pars: wire parameter dict. entry for the wire.

    Returns
    -------
        Coordinates of closest approach along the wire.
    """
    # get the wire unit and coordinate vectors
    e_wire = np.asarray(wire_pars["e"])
    r_wire = np.asarray(wire_pars["r0"])
        
    p_track = np.asarray(track_pars[0])
    # compute the track unit-vector
    e_track = np.array(
        [
            mt.cos(track_pars[1]) * mt.cos(track_pars[2]),
            mt.cos(track_pars[1]) * mt.sin(track_pars[2]),
            mt.sin(track_pars[1]),
        ]
    )
    if np.any(np.dot(e_track, e_wire) != 1):
        w_len = (
            np.dot((r_wire - p_track), (e_wire - e_track * np.dot(e_track, e_wire)))
        ) / ((np.dot(e_track, e_wire)) ** 2 - 1)
    else:
        w_len = 0
    return w_len

def track_plane_intersec(track_pars, plane_pars) -> np.ndarray:
    """Computes the intersection between plane and track.

    Args:
        track_pars: list of track parameters [[p0],theta,phi]. Angles in radians.
        plane_pars: list of plane parameters in the [[p0],[norm_vers]] format.

    Returns: the intersection point coordinates as a numpy vector.
    """
    # get the p0 and versors as numpy arrays
    p_plane = np.asarray(plane_pars[0])
    e_plane = np.asarray(plane_pars[1])
    p_track = np.asarray(track_pars[0])
    e_track = np.array(
        [
            mt.cos(track_pars[1]) * mt.cos(track_pars[2]),
            mt.cos(track_pars[1]) * mt.sin(track_pars[2]),
            mt.sin(track_pars[1]),
        ]
    )

    # compute the intersection point
    if (np.dot(e_track, e_plane)) != 0:
        return p_track + e_track * (np.dot((p_plane - p_track), e_plane)) / (
            np.dot(e_track, e_plane)
        )
    # return an out of bounds point if track and plane are orthogonal
    else:
        return np.array([1e4, 1e4, 1e4])

def point_wire_distance(point_par, wire_pars) -> float:
    """Computes the point-wire distance given their parameters.

    Args:
        point_par: point coordinates.
        wire_pars: wire parameters dict.
    Returns: the point-wire distance.
    """
    return np.linalg.norm(np.cross((point_par - wire_pars["r0"]), wire_pars["e"]))

def simulate_time(track_wire_distance, wire_length, smearing=True):
    """
    Simulates the signal time from the track distance adding instrumental effects.

    Args
    ----
        track_wire_distance: track-wire distance.
        wire_intercept: min. track distance position along the wire.
        smearing: bool for whether or not to add gaussian smearing.

    Returns
    -------
    The simulated signal time [ns]. Track is assumed to be istantaneous.
    """
    # simulate a signal time given the drift velocity, intercept on the wire + gaussian smearing
    if smearing:
        return (
            track_wire_distance / __v_drift__
            + wire_length / __v_signal__
            + np.random.normal(0, 4.0)
        )
    else:
        return track_wire_distance / __v_drift__ + wire_length / __v_signal__