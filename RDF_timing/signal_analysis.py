# Import ROOT modules
from ROOT import gInterpreter, TChain, TFile, RDataFrame, std

# Import Python libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

# Custom modules import
import FunctionDefines as rdf_functions
import Utils
import Wavelets


# Wrapper for argument parsing
def ParseArgs():
    """Set arguments for running on command line"""
    parser = argparse.ArgumentParser("Signal analysis parameter selection.")
    parser.add_argument(
        "--n_cores",
        required=False,
        default=0,
        help="Number of threads (cores) set for the job",
        type=int,
    )
    parser.add_argument(
        "--channel_name", required=False, default="v1752_ch00", type=str
    )


if __name__ == "__main__":

    # ROOT.EnableImplicitMT()
    # n_threads = ROOT.GetThreadPoolSize()
    # print("Working with: ", n_threads, " threads")

    # Declaration of C++ code
    gInterpreter.Declare(rdf_functions.GetCppCode())

    ##% Read digitizer and track reconstruction data files
    # Read the v1751 files in the right order
    v1751_TChain = TChain("EventsFromDigitizer")
    for idx in range(60, 188):
        v1751_TChain.Add("./data_files/run_{0}_0x17510000v1751.root".format(idx))
    v1751_TChain.SetName("v1751")

    # Read the v1752 files in the right order
    v1752_TChain = TChain("EventsFromDigitizer")
    for idx in range(60, 188):
        v1752_TChain.Add("./data_files/run_{0}_0x17520000v1751.root".format(idx))
    v1752_TChain.SetName("v1752")

    # Impose that the chains have the same number of entries
    assert (
        v1751_TChain.GetEntries() == v1752_TChain.GetEntries()
    ), "The chains have different lengths :("

    # Join the TChains as friends if their length is the same
    v1751_TChain.AddFriend(v1752_TChain)
    print("TChain entries: ", v1751_TChain.GetEntries())

    # Process the raw track file if not yet done
    # Utils.process_tracks_file("./data_files/tracks.root", v1751_TChain.GetEntries())

    # read the track parameter tree
    track_file = TFile.Open("./data_files/out_track_tree.root", "READ")
    track_tree = track_file.tr_pars

    # add the track parameter file as friend
    assert (
        v1751_TChain.GetEntries() == track_tree.GetEntries()
    ), "The chains have different lengths :("
    v1751_TChain.AddFriend(track_tree)

    t_start = time.time()
    # initialize the RDF from the TChains
    sx_DFrame = RDataFrame(v1751_TChain)
    # Get the list of channel names from both TChains
    col_dict = np.array(sx_DFrame.GetColumnNames(), dtype=object)[np.r_[:8, 16:24]]

    # Define a copy of the RDF on which to apply computations
    sx_transform = sx_DFrame

    # Apply waveform computations to all channels
    for col_name in col_dict:
        digi_name = "v1751" if len(col_name) == 4 else "v1752"
        ch_name = str(col_name)[-4:]
        sx_transform = (
            sx_transform.Define(
                "{0}_{1}_SB".format(digi_name, ch_name),
                "Numba::d_SubtractBaseline({0}, 500)".format(col_name),
            )
            .Define(
                "{0}_{1}_FFTmag".format(digi_name, ch_name),
                "d_ComputeFFT({0}_{1}_SB)".format(digi_name, ch_name),
            )
            .Define(
                "{0}_{1}_WF".format(digi_name, ch_name),
                "Numba::d_HaarWaveletFilter({0}_{1}_SB,1e-2)".format(
                    digi_name, ch_name
                ),
            )
            .Define(
                "{0}_{1}_WFirm".format(digi_name, ch_name),
                "Numba::d_HaarWaveletFirm({0}_{1}_SB,8e-3,5e-2)".format(
                    digi_name, ch_name
                ),
            )
            .Define(
                "{0}_{1}_WF_int".format(digi_name, ch_name),
                "Numba::d_GetTotInt({0}_{1}_WFirm,0.02)".format(digi_name, ch_name),
            )
            .Define(
                "{0}_{1}_SlideInt".format(digi_name, ch_name),
                "Numba::d_WfSlideInt({0}_{1}_SB,16)".format(digi_name, ch_name),
            )
        )
    t_end = time.time()
    print("RDF definition lasted: ", t_end - t_start, " s")

    t_start = time.time()

    # fill the list of columns to be saved to file
    col_vec = std.vector("std::string")()
    for digi_name in ["v1751", "v1752"]:
        for ch_name in ["ch0{0}".format(i) for i in range(8)]:
            col_vec.push_back("{0}_{1}_SB".format(digi_name, ch_name))
            if ch_name not in [
                "ch06",
                "ch07",
            ]:
                col_vec.push_back("{0}_{1}_FFTmag".format(digi_name, ch_name))
                col_vec.push_back("{0}_{1}_WFirm".format(digi_name, ch_name))
                col_vec.push_back("{0}_{1}_WF_int".format(digi_name, ch_name))
    for col in ["tr_pars.r0_x", "tr_pars.r0_y", "tr_pars.sx", "tr_pars.sy"]:
        col_vec.push_back(col)

    range_lst = np.append(
        np.arange(0, v1751_TChain.GetEntries(), 20000), v1751_TChain.GetEntries()
    ).tolist()
    print(range_lst)
    for idx in range(len(range_lst) - 1):
        print("Evts: ", range_lst[idx], "-", range_lst[idx + 1])
        sub = sx_transform.Range(range_lst[idx], range_lst[idx + 1])
        sub.Snapshot(
            "ProcWfTree", "./data_files/processed_wfs_WFirm_{0}.root".format(idx), col_vec
        )

    # # Process a small subset of events
    # sub = sx_transform.Filter(
    #     "Numba::f_ValAboveThr(v1752_ch00_WF_int,{0}) && Numba::f_FFTInRange(v1752_ch00_FFTmag, 3,0.,0.4)".format(
    #         0.005 * 3
    #     )
    # ).Range(0, 20000)
    # sub_dict = sub.AsNumpy(["v1752_ch00_SB","v1752_ch00_WFirm"])
    # t_end = time.time()
    # print("Computation lasted: ", t_end - t_start, " s")

    # t_srsSB = np.arange(np.asarray(sub_dict["v1752_ch00_SB"][0]).shape[0]) * 1e-9
    # t_srsWF = np.arange(np.asarray(sub_dict["v1752_ch00_WFirm"][0]).shape[0]) * 1e-9

    # np.save("beat.npy",sub_dict["v1752_ch00_SB"][14])
    # for idx in range(10, 30):
    #     fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
    #     plt.title("WF-Waveforms and $t_{10\%}$ (v1752_ch00)", fontsize=18)
    #     plt.xlabel("Time [ns]", fontsize=14)
    #     plt.ylabel("Amp [V]", fontsize=14)
    #     ax.minorticks_on()
    #     plt.grid()
    #     plt.tight_layout()
    #     plt.plot(
    #         t_srsSB,
    #         sub_dict["v1752_ch00_SB"][idx],
    #         markersize=2,
    #         label="SB-Waveform",
    #     )
    #     plt.plot(
    #         t_srsWF,
    #         sub_dict["v1752_ch00_WFirm"][idx],
    #         markersize=2,
    #         label="WF-Waveform",
    #     )
    #     plt.legend(fontsize=16)
    #     plt.show()
