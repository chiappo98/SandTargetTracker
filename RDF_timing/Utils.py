##% A repository of utility functions for the prototype data analysis

# library imports
import ROOT
import numpy as np
import array as ar

from pyparsing import identbodychars
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

##% Read the tracks.root file and extract the dictionary of parameters to be added to the RDF


def process_tracks_file(track_file_path, rdf_size):
    """Read and process the track parameters .root file to dictionary that can be read through RDF.
        Missing track entries are padded.

    Args:
        track_file_path (str): the path of the track.root file.
        rdf_size (int): size of the RDF to merge with.

    Returns:
        dict: a dictionary containing the numpy arrays of track parameters.
    """
    # read the track parametrer file tree as an RDF
    track_file = ROOT.TFile.Open(track_file_path, "READ")
    track_df = ROOT.RDataFrame(track_file.tree)
    # extract the dictionary of the parameters (as np.arrays)
    columns_dict = track_df.AsNumpy(["entry", "x", "y", "sx", "sy"])
    np_entry = columns_dict["entry"]

    # the track file may have more entries that the RDF: slice down the arrays in such a case
    max_idx = -1
    if np_entry.shape[0] > rdf_size:
        max_idx = np.argwhere(np_entry == rdf_size)[0, 0]
    else:
        print("RDF and track file have the same shape.")
    # change the entries to rdf_size-sized arrays with a flag value for the empty events
    for key, array in columns_dict.items():
        base_array = np.full(rdf_size, -999.0)
        # This seems a bug!
        base_array[np_entry] = array
        columns_dict[key] = base_array

    # remove the entries column and change some names for better readability
    columns_dict.pop("entry")
    columns_dict["r0_x"] = columns_dict.pop("x")
    columns_dict["r0_y"] = columns_dict.pop("y")

    # Fill a TTree with the content of the dictionary
    out_file_name = track_file_path.rpartition("/")[0] + "/out_track_tree.root"
    track_file = ROOT.TFile.Open(out_file_name, "RECREATE")
    track_tree = ROOT.TTree("tr_pars", "Track Parameters Tree")

    # declare vars to be read when filling
    var_rx = ar.array("d", [0])
    var_ry = ar.array("d", [0])
    var_sx = ar.array("d", [0])
    var_sy = ar.array("d", [0])

    # define the TTree branches
    track_tree.Branch("r0_x", var_rx, "r0_x/D")
    track_tree.Branch("r0_y", var_ry, "r0_y/D")
    track_tree.Branch("sx", var_sx, "sx/D")
    track_tree.Branch("sy", var_sy, "sy/D")

    # fill the tree entries
    for idEntry in range(columns_dict["r0_x"].shape[0]):
        var_rx[0] = columns_dict["r0_x"][idEntry]
        var_ry[0] = columns_dict["r0_y"][idEntry]
        var_sx[0] = columns_dict["sx"][idEntry]
        var_sy[0] = columns_dict["sy"][idEntry]
        track_tree.Fill()

    track_tree.Write()

    return out_file_name


def GetPCAParameters(proj_df):
    np_XY_proj = proj_df.AsNumpy(["X_proj", "Y_proj"])

    np_XY_proj = np.column_stack((np_XY_proj["X_proj"], np_XY_proj["Y_proj"]))

    hdb = HDBSCAN(
        min_cluster_size=100, cluster_selection_epsilon=1.5, allow_single_cluster=True
    )
    hdb.fit(np_XY_proj)

    # fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
    # plt.axis("equal")
    # plt.title("PCA on the clustered wire hits", fontsize=18)
    # plt.xlabel("$X_{reco}$ [cm]", fontsize=18)
    # plt.ylabel("$Y_{reco}$ [cm]", fontsize=18)
    # ax.minorticks_on()
    # plt.scatter(
    #     np_XY_proj[:, 0][hdb.labels_ == -1],
    #     np_XY_proj[:, 1][hdb.labels_ == -1],
    #     s=2,
    #     c="lightslategray",
    # )
    # plt.scatter(
    #     np_XY_proj[:, 0][hdb.labels_ == 0],
    #     np_XY_proj[:, 1][hdb.labels_ == 0],
    #     s=2,
    #     c="deepskyblue",
    #     label="ch_00 cluster",
    # )

    np_XY_proj = np_XY_proj[hdb.labels_ == 0, :]
    pca = PCA(n_components=2).fit(np_XY_proj)
    return pca.mean_, pca.components_


##% MAIN for testing
if __name__ == "__main__":
    pars_dict = process_tracks_file("./data_files/tracks.root", 127766)

    # track_file = ROOT.TFile.Open("./data_files/tracks.root", "READ")
    # track_tree = track_file.tree
    # print(track_tree.Show(0))
