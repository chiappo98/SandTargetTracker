# library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

# import the wire dictionary maker 
from common_functions import make_config_dictionary

##% Analysis scripts
def check_deviations(evt_df, wire_dict, save_path):
    true_pos = np.array([evt[0] for evt in evt_df["track_pars"].values])
    fit_pos = np.array([evt[0] for evt in evt_df["fit_pars"].values])
    reco_dist = np.linalg.norm(fit_pos - true_pos, axis=1)
    thr = 2e-2
    n_above_thr = (reco_dist > thr).sum()
    print("-- Checking fit deviations --")
    # log the number of deviations above threshold
    print("{0}/{1} evts. have Delta > {2}".format(n_above_thr, reco_dist.shape[0], thr))

    # geometric distribution of the deviations
    fig_dev = plt.figure(figsize=(11, 10),tight_layout=True).add_subplot()
    plt.scatter(
        true_pos[:, 0],
        true_pos[:, 1],
        c=reco_dist,
        label="$\Delta_r$ vs. $(x,y)_{true}$ ",
        s=6,
        marker="s",
        cmap="plasma",
    )
    plt.title(
        r"$\Delta_r$ vs. $(x,y)_{true},\quad \theta \equiv \varphi \equiv 0^\circ$",
        fontsize=16,
    )
    plt.xlabel("$x_{true}$  [cm]", fontsize=14)
    plt.ylabel("$y_{true}$  [cm]", fontsize=14)
    cb = plt.colorbar()
    cb.ax.set_title("$\Delta_r$ [cm]")

    # superimpose wires to the plot
    # set the initial and final drawing coordinates
    coor_lsp = np.array([4, 26.0])
    for wire_key, wire in wire_dict["wires"].items():
        x = wire["r0"][0] + (coor_lsp /  wire["e"][1]) *  wire["e"][0]
        x_f = wire["r0"][0] + (coor_lsp / wire["e"][1]) * wire["e"][0] - 1.0
    plt.savefig("{0}/Delta_truexy.pdf".format(save_path))



def deviations_histo(true_pos, fit_pos, wire_dict, save_path):
    # same as before: get the coordinates and the reconstructed distance
    reco_dist = np.linalg.norm(fit_pos - true_pos, axis=1)
    thr = 1e-4
    n_above_thr = (reco_dist > thr).sum()
    print("-- Checking fit deviations --")
    # log the number of deviations above threshold
    print("{0}/{1} evts. have Delta > {2}".format(n_above_thr, reco_dist.shape[0], thr))

    x_bins, x_step = np.linspace(-1, 23, 200, retstep=True)
    y_bins, y_step = np.linspace(0, 30, 250, retstep=True)

    mean_array = np.empty((0))
    for id_x in range(x_bins.shape[0]):
        for id_y in range(y_bins.shape[0]):
            bin_dists = reco_dist[
                np.argwhere(
                    (x_bins[id_x] <= true_pos[:, 0])
                    & (true_pos[:, 0] < x_bins[id_x] + x_step)
                    & (y_bins[id_y] <= true_pos[:, 1])
                    & (true_pos[:, 1] < y_bins[id_y] + y_step)
                )
            ]
            if bin_dists.shape[0] > 0:
                mean_array = np.append(
                    mean_array, np.sum(bin_dists) / bin_dists.shape[0]
                )
            else:
                mean_array = np.append(mean_array, 0.0)

    fig_dev = plt.figure(figsize=(11, 10),tight_layout=True).add_subplot()

    c_mat = np.array(np.meshgrid(x_bins, y_bins)).T.reshape(-1, 2)
    plt.hist2d(
        x=c_mat[:, 0],
        y=c_mat[:, 1],
        bins=x_bins.shape[0],
        weights=mean_array,
        label="$\Delta_r$ vs. $(x,y)_{true}$ ",
        cmap="plasma",
    )
    plt.title(
        r"$\langle\Delta_r\rangle$ vs. $(x,y)_{true},\quad \theta \equiv \varphi \equiv 0^\circ$",
        fontsize=20,
    )
    plt.xlabel("$x_{true}$  [cm]", fontsize=18)
    plt.ylabel("$y_{true}$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    cb.ax.set_title(r"$\langle\Delta_r\rangle}$ [cm]", fontsize=18)

    # superimpose wires to the plot
    # set the initial and final drawing coordinates
    coor_lsp = np.array([0, 30.0])
    for wire_key, wire in wire_dict["wires"].items():
        x = wire["r0"][0] + (coor_lsp /  wire["e"][1]) *  wire["e"][0]
        x_f = wire["r0"][0] + (coor_lsp / wire["e"][1]) * wire["e"][0] - 1.0
        fig_dev.plot(x, coor_lsp, color="tab:cyan")
        fig_dev.plot(x_f, coor_lsp, color="tab:red")
    plt.savefig("{0}/Delta_truexy_histo.pdf".format(save_path))

def max_deviation_histo(true_pos, fit_pos, wire_dict, save_path):
    # same as before: get the coordinates and the reconstructed distance
    reco_dist = np.linalg.norm(fit_pos - true_pos, axis=1)
    thr = 2e-2
    n_above_thr = (reco_dist > thr).sum()
    print("-- Checking fit deviations --")
    # log the number of deviations above threshold
    print("{0}/{1} evts. have Delta > {2}".format(n_above_thr, reco_dist.shape[0], thr))

    x_bins, x_step = np.linspace(-1, 23, 200, retstep=True)
    y_bins, y_step = np.linspace(0, 30, 250, retstep=True)

    max_array = np.empty((0))
    for id_x in range(x_bins.shape[0]):
        for id_y in range(y_bins.shape[0]):
            bin_dists = reco_dist[
                np.argwhere(
                    (x_bins[id_x] <= true_pos[:, 0])
                    & (true_pos[:, 0] < x_bins[id_x] + x_step)
                    & (y_bins[id_y] <= true_pos[:, 1])
                    & (true_pos[:, 1] < y_bins[id_y] + y_step)
                )
            ]
            if bin_dists.shape[0] > 0:
                max_array = np.append(
                    max_array, np.max(bin_dists)
                )
            else:
                max_array = np.append(max_array, 0.0)

    fig_dev = plt.figure(figsize=(11, 10),tight_layout=True).add_subplot()

    c_mat = np.array(np.meshgrid(x_bins, y_bins)).T.reshape(-1, 2)
    plt.hist2d(
        x=c_mat[:, 0],
        y=c_mat[:, 1],
        bins=x_bins.shape[0],
        weights=max_array,
        label="$\Delta_r$ vs. $(x,y)_{true}$ ",
        cmap="plasma",
        norm = colors.LogNorm()
    )
    plt.title(
        r"Max$(\Delta_r)$ vs. $(x,y)_{true},\quad \theta \equiv \varphi \equiv 0^\circ$",
        fontsize=20,
    )
    plt.xlabel("$x_{true}$  [cm]", fontsize=18)
    plt.ylabel("$y_{true}$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    cb.ax.set_title(r"Max$(\Delta_r)$ [cm]", fontsize=18)

    # superimpose wires to the plot
    # set the initial and final drawing coordinates
    coor_lsp = np.array([0, 30.0])
    for wire_key, wire in wire_dict["wires"].items():
        x = wire["r0"][0] + (coor_lsp /  wire["e"][1]) *  wire["e"][0]
        x_f = wire["r0"][0] + (coor_lsp / wire["e"][1]) * wire["e"][0] - 1.0
        fig_dev.plot(x, coor_lsp, color="tab:cyan")
        fig_dev.plot(x_f, coor_lsp, color="tab:red")
    plt.savefig("{0}/Max_delta_truexy_histo.pdf".format(save_path))

def true_vs_fit_coords(true_pos, fit_pos, save_path):
    # true vs. fit coordinate plots.
    fig_xtd = plt.figure(figsize=(11, 10),tight_layout=True)
    plt.hist2d(true_pos[:,0], fit_pos[:,0],bins=(100,100),cmap="plasma",norm=colors.PowerNorm(gamma=0.5))
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    cb.ax.set_title("Counts", fontsize=18)
    plt.title(
        r"$x_{true}\,vs.\,x_{fit},\quad \theta \equiv \varphi \equiv 0^\circ$", fontsize=20
    )
    plt.xlabel("$x_{true}$  [cm]", fontsize=18)
    plt.ylabel("$x_{fit}$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("{0}/xt_xf_t0p0.pdf".format(save_path))

    fig_xtd = plt.figure(figsize=(11, 10),tight_layout=True)
    plt.hist2d(true_pos[:,1], fit_pos[:,1],bins=(100,100),cmap="plasma",norm=colors.PowerNorm(gamma=0.5))
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    cb.ax.set_title("Counts", fontsize=18)
    plt.title(
        r"$y_{true}\,vs.\,y_{fit},\quad \theta \equiv \varphi \equiv 0^\circ$", fontsize=20
    )
    plt.xlabel("$y_{true}$  [cm]", fontsize=18)
    plt.ylabel("$y_{fit}$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("{0}/yt_yf_t0p0.pdf".format(save_path))


def abs_ghosts(true_pos, fit_pos, wire_dict, save_path):
    # get the vector of true-fit distances
    reco_dist = np.linalg.norm(fit_pos - true_pos, axis=1)
    delta_thr = 1e-1
    dev_true_x = true_pos[:,0][reco_dist > delta_thr]
    dev_fit_x = fit_pos[:,0][reco_dist > delta_thr]
    # now get the corresponding ys
    dev_true_y = true_pos[:,1][reco_dist > delta_thr]
    dev_fit_y = fit_pos[:,1][reco_dist > delta_thr]

    # indicatively the number of such ghosts is:
    print(
        "Delta > 2e-2 points: ",
        dev_true_x.shape[0],
        "/",
        true_pos.shape[0],
        " of the total.",
    )
    # get the true coordinates of the points above threshold on x
    fig_xtd = plt.figure(figsize=(10, 10),tight_layout=True).add_subplot()
    plt.scatter(dev_true_x, dev_true_y, c="steelblue", s=4, label="Point $(x,y)_{true}$")
    plt.scatter(dev_fit_x, dev_fit_y, c="orangered", s=4, label="Point $(x,y)_{fit}$")
    plt.title(
        r"Point $(x,y),\quad\Delta_r>1\cdot 10^{-1}\, cm,\quad \theta \equiv \varphi \equiv 0^\circ$",
        fontsize=20,
    )
    plt.xlabel("$x$  [cm]", fontsize=18)
    plt.ylabel("$y$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    coor_lsp = np.array([0, 30.0])
    for wire_key, wire in wire_dict["wires"].items():
        x = wire["r0"][0] + (coor_lsp /  wire["e"][1]) *  wire["e"][0]
        fig_xtd.plot(x, coor_lsp, color="forestgreen")
    plt.xlim(4, 20)
    plt.savefig("{0}/abs_ghosts_check.pdf".format(save_path))

    # # get the fit coordinates of the points above threshold on x
    fig_xtd = plt.figure(figsize=(10, 10),tight_layout=True).add_subplot()
    plt.title(
        r"$(x,y)_{fit},\quad\Delta_r>1\cdot 10^{-1}\, cm,\quad \theta \equiv \varphi \equiv 0^\circ$", fontsize=20
    )
    for i in range(500):
        plt.arrow(dev_true_x[i],dev_true_y[i],(dev_fit_x[i]-dev_true_x[i]),(dev_fit_y[i]-dev_true_y[i]),head_width=1e-1,color='orangered')
    for wire_key, wire in wire_dict["wires"].items():
        x = wire["r0"][0] + (coor_lsp /  wire["e"][1]) *  wire["e"][0]
        fig_xtd.plot(x, coor_lsp, color="forestgreen")
    plt.xlim(4, 20)
    plt.xlabel("$x$  [cm]", fontsize=18)
    plt.ylabel("$y$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("{0}/abs_ghosts_check_arrow.pdf".format(save_path))

def y_ghosts(true_pos, fit_pos, wire_dict, save_path):
    # Check points with Delta_y above some threshold
    # With a delta_thr > 2e-0 cut, there remain the x-ghosts (above) and the points along the bisector line of the stereo wires
    delta_thr = 4e-2

    dev_true_x = true_pos[:,0][abs(true_pos[:,1] - fit_pos[:,1]) > delta_thr]
    dev_fit_x = fit_pos[:,0][abs(true_pos[:,1] - fit_pos[:,1]) > delta_thr]
    # now get the corresponding ys
    dev_true_y = true_pos[:,1][abs(true_pos[:,1] - fit_pos[:,1]) > delta_thr]
    dev_fit_y = fit_pos[:,1][abs(true_pos[:,1] - fit_pos[:,1]) > delta_thr]

    # indicatively the number of such ghosts is:
    print(
        "Y-wise ghosts: ",
        dev_true_x.shape[0],
        "/",
        true_pos.shape[0],
        " of the total.",
    )

    # get the true coordinates of the points above threshold on x
    fig_xtd = plt.figure(figsize=(10, 10),tight_layout=True).add_subplot()
    plt.scatter(dev_true_x, dev_true_y, c="steelblue", s=4, label="Point $(x,y)_{true}$")
    plt.scatter(dev_fit_x, dev_fit_y, c="orangered", s=4, label="Point $(x,y)_{fit}$")
    plt.title(
        r"Point $(x,y),\quad|y_{true}-y_{fit}|>4\cdot 10^{-2}\, cm,\quad \theta \equiv \varphi \equiv 0^\circ$",
        fontsize=20,
    )
    plt.xlabel("$x$  [cm]", fontsize=18)
    plt.ylabel("$y$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    coor_lsp = np.array([0, 30.0])
    for wire_key, wire in wire_dict["wires"].items():
        x = wire["r0"][0] + (coor_lsp /  wire["e"][1]) *  wire["e"][0]
        fig_xtd.plot(x, coor_lsp, color="forestgreen")
    plt.xlim(4, 20)
    plt.savefig("{0}/y_ghosts_check.pdf".format(save_path))

    # # get the fit coordinates of the points above threshold on x
    fig_xtd = plt.figure(figsize=(10, 10),tight_layout=True).add_subplot()
    plt.title(
        r"$(x,y)_{fit},\quad|y_{true}-y_{fit}|>4\cdot 10^{-2}\, cm,\quad \theta \equiv \varphi \equiv 0^\circ$", fontsize=20
    )
    for i in range(500):
        plt.arrow(dev_true_x[i],dev_true_y[i],(dev_fit_x[i]-dev_true_x[i]),(dev_fit_y[i]-dev_true_y[i]),head_width=1e-1,color='orangered')
    for wire_key, wire in wire_dict["wires"].items():
        x = wire["r0"][0] + (coor_lsp /  wire["e"][1]) *  wire["e"][0]
        fig_xtd.plot(x, coor_lsp, color="forestgreen")
    plt.xlim(4, 20)
    plt.xlabel("$x$  [cm]", fontsize=18)
    plt.ylabel("$y$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("{0}/y_ghosts_check_arrow.pdf".format(save_path))

def x_ghosts(true_pos, fit_pos, wire_dict, save_path):
    delta_thr = 4e-2

    dev_true_x = true_pos[:,0][abs(true_pos[:,0] - fit_pos[:,0]) > delta_thr]
    dev_fit_x = fit_pos[:,0][abs(true_pos[:,0] - fit_pos[:,0]) > delta_thr]
    # now get the corresponding ys
    dev_true_y = true_pos[:,1][abs(true_pos[:,0] - fit_pos[:,0]) > delta_thr]
    dev_fit_y = fit_pos[:,1][abs(true_pos[:,0] - fit_pos[:,0]) > delta_thr]

    # indicatively the number of such ghosts is:
    print(
        "X-wise ghosts: ",
        dev_true_x.shape[0],
        "/",
        true_pos.shape[0],
        " of the total.",
    )
    # get the true coordinates of the points above threshold on x
    fig_xtd = plt.figure(figsize=(10, 10),tight_layout=True).add_subplot()
    plt.scatter(dev_true_x, dev_true_y, c="steelblue", s=4, label="Point $(x,y)_{true}$")
    plt.scatter(dev_fit_x, dev_fit_y, c="orangered", s=4, label="Point $(x,y)_{fit}$")
    plt.title(
        r"Point $(x,y),\quad|x_{true}-x_{fit}|>4\cdot 10^{-2}\, cm,\quad \theta \equiv \varphi \equiv 0^\circ$",
        fontsize=20,
    )
    plt.xlabel("$x$  [cm]", fontsize=18)
    plt.ylabel("$y$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    coor_lsp = np.array([0, 30.0])
    for wire_key, wire in wire_dict["wires"].items():
        x = wire["r0"][0] + (coor_lsp /  wire["e"][1]) *  wire["e"][0]
        fig_xtd.plot(x, coor_lsp, color="forestgreen")
    plt.xlim(4, 20)
    plt.savefig("{0}/x_ghosts_check.pdf".format(save_path))

    # # get the fit coordinates of the points above threshold on x
    fig_xtd = plt.figure(figsize=(10, 10),tight_layout=True).add_subplot()
    plt.title(
        r"$(x,y)_{fit},\quad|x_{true}-x_{fit}|>4\cdot 10^{-2}\, cm,\quad \theta \equiv \varphi \equiv 0^\circ$", fontsize=20
    )
    for i in range(500):
        plt.arrow(dev_true_x[i],dev_true_y[i],(dev_fit_x[i]-dev_true_x[i]),(dev_fit_y[i]-dev_true_y[i]),head_width=1e-1,color='orangered')
    for wire_key, wire in wire_dict["wires"].items():
        x = wire["r0"][0] + (coor_lsp /  wire["e"][1]) *  wire["e"][0]
        fig_xtd.plot(x, coor_lsp, color="forestgreen")
    plt.xlim(4, 20)
    plt.xlabel("$x$  [cm]", fontsize=18)
    plt.ylabel("$y$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("{0}/x_ghosts_check_arrow.pdf".format(save_path))

def range_y_ghosts(true_pos, fit_pos, wire_dict, save_path):
    # Check points with delta_y within some threshold
    delta_thr_l = 2e-1
    delta_thr_u = 1e-0
    dev_true_x = true_pos[:,0][
        np.logical_and(delta_thr_l < abs(true_pos[:,1] - fit_pos[:,1]), abs(true_pos[:,1] - fit_pos[:,1]) < delta_thr_u)
    ]
    dev_fit_x = fit_pos[:,0][np.logical_and(delta_thr_l < abs(true_pos[:,1] - fit_pos[:,1]), abs(true_pos[:,1] - fit_pos[:,1]) < delta_thr_u)]
    # now get the corresponding ys
    dev_true_y = true_pos[:,1][np.logical_and(delta_thr_l < abs(true_pos[:,1] - fit_pos[:,1]), abs(true_pos[:,1] - fit_pos[:,1]) < delta_thr_u)]
    dev_fit_y = fit_pos[:,1][np.logical_and(delta_thr_l < abs(true_pos[:,1] - fit_pos[:,1]), abs(true_pos[:,1] - fit_pos[:,1]) < delta_thr_u)]

    # indicatively the number of such ghosts is:
    print(
        "Vertical sense wire ghosts: ",
        dev_true_x.shape[0],
        "/",
        true_pos.shape[0],
        " of the total.",
    )

    # get the true coordinates of the points above threshold on x
    fig_xtd = plt.figure(figsize=(10, 10),tight_layout=True).add_subplot()
    coor_lsp = np.array([0, 30.0])
    for wire_key, wire in wire_dict["wires"].items():
        x = wire["r0"][0] + (coor_lsp /  wire["e"][1]) *  wire["e"][0]
        x_f = wire["r0"][0] + (coor_lsp / wire["e"][1]) * wire["e"][0] - 1.0
        fig_xtd.plot(x, coor_lsp, color="forestgreen")
        fig_xtd.plot(x_f, coor_lsp, color="gold")
    plt.scatter(dev_true_x, dev_true_y, c="steelblue", s=4, label="Point $(x,y)_{true}$")
    plt.scatter(dev_fit_x, dev_fit_y, c="orangered", s=4, label="Point $(x,y)_{fit}$")
    plt.title(
        r"Point $(x,y),\quad 0.2<|y_{true}-y_{fit}|>1\, cm,\quad \theta \equiv \varphi \equiv 0^\circ$",
        fontsize=20,
    )
    plt.xlabel("$x$  [cm]", fontsize=18)
    plt.ylabel("$y$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.xlim(4, 20)
    plt.xticks(np.arange(4,21,1))
    plt.yticks(np.arange(1,30,1))
    plt.grid(which='both',linestyle='--')
    plt.savefig("{0}/y_range_ghosts.pdf".format(save_path))

    # # get the fit coordinates of the points above threshold on x
    fig_xtd = plt.figure(figsize=(10, 10),tight_layout=True).add_subplot()
    plt.title(
        r"$(x,y)$ displacement,$\quad 0.2<|y_{true}-y_{fit}|>1\, cm,\quad \theta \equiv \varphi \equiv 0^\circ$",
        fontsize=20,
    )
    for wire_key, wire in wire_dict["wires"].items():
        x = wire["r0"][0] + (coor_lsp /  wire["e"][1]) *  wire["e"][0]
        x_f = wire["r0"][0] + (coor_lsp / wire["e"][1]) * wire["e"][0] - 1.0
        fig_xtd.plot(x, coor_lsp, color="forestgreen")
        fig_xtd.plot(x_f, coor_lsp, color="gold")
    for i in range(dev_true_x.shape[0]):
        plt.arrow(
            dev_true_x[i],
            dev_true_y[i],
            (dev_fit_x[i] - dev_true_x[i]),
            (dev_fit_y[i] - dev_true_y[i]),
            head_width=1e-1,
            color="orangered",
        )
    plt.xlim(4, 20)
    plt.xlabel("$x$  [cm]", fontsize=18)
    plt.ylabel("$y$  [cm]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(np.arange(4,21,1))
    plt.yticks(np.arange(1,30,1))
    plt.grid(which='both',linestyle='--')
    plt.savefig("{0}/y_range_ghosts_arrow.pdf".format(save_path))

##% Utils
def merge_dataframes(dfs_folder):
    """Merges all the dataframe chunks contained in the specified folder.

        Args:
            dfs_folder: path containing the dataframe chunks. Requires a '/' at the end of the string.
        Returns: the final merged dataframe.
    """
    # get the list of dataframe chunks in the folder
    df_paths = [dfs_folder + f for f in os.listdir(dfs_folder) if f.endswith(".pkl")]
    # merge all dataframes in the selected folder
    return pd.concat(map(pd.read_pickle,df_paths))

##% MAIN
if __name__ == "__main__":

    # set the container folder
    dfs_folder = "/storage/gpfs_data/neutrino/users/alrugger/Software/dchamber_prototype_analysis/dataframes/fitted/nev1e5_ncall2e3_smear4ns/"
    # get the merged dataframe
    merged_df  = merge_dataframes(dfs_folder)
    print("Merged df. length: ",len(merged_df))

    # get the wire configuration dictionary
    wire_dict = make_config_dictionary()

    # get the arrays of true and fitted coordinates
    true_pos = np.array([evt[0] for evt in merged_df["track_pars"].values])
    fit_pos = np.array([evt[0] for evt in merged_df["fit_pars"].values])

    # set the save path for the plots
    save_path = "../plots/plots_ncall2e3_smear4ns"
    # create the save folder if t doesn't exist already
    if not os.path.exists(save_path):
        print("> Creating new output folder")
        os.makedirs(save_path)
    else:
        print("> Output folder already exists")

    # plot the mean abs-deviations from the true position
    #check_deviations(merged_df,wire_dict,save_path)

    # plot the mean abs-deviations from the true position as an histogram
    deviations_histo(true_pos,fit_pos,wire_dict,save_path)
        
    # # plot the max abs-deviations from the true position as an histogram
    # max_deviation_histo(true_pos,fit_pos,wire_dict,save_path)

    # # plot the true vs. fit coordinates
    # true_vs_fit_coords(true_pos,fit_pos,save_path)

    # plot the points with Delta_r > 2e-2
    abs_ghosts(true_pos,fit_pos, wire_dict, save_path)

    # # plot the x-wise ghost points
    # x_ghosts(true_pos,fit_pos, wire_dict, save_path)

    # # plot the y-wise ghost points
    # y_ghosts(true_pos,fit_pos, wire_dict, save_path)
        
    # # plot the ghosts with y-within a range
    # range_y_ghosts(true_pos,fit_pos, wire_dict, save_path)
