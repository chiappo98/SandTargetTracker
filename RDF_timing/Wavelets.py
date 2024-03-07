##% Where I try to write Wavelet filters for processing the chamber signals
from re import A
import ROOT
import numpy as np
import pywt
from numba.typed import List
from numba import njit
import matplotlib.pyplot as plt


##% Define wavelet and scaling functions (high-pass and low-pass filters)
@njit
def HaarFunctions():
    """Returns the wavelet and scaling functions of the Haar wavelet.

    Returns:
        ndarray: arrays of the wavelet and scaling function respectively.
    """
    # high-pass/wavelet (diff.), low-pass/scaling (sum)
    return np.array([1.0, -1.0]), np.array([1.0, 1.0])


@njit(fastmath=True)
def nb_convolve(a, b) -> np.ndarray:
    return np.asarray(np.convolve(a, b))


##% Thresholding

@njit
def Threshold(coeffs, thr_val, mode="soft") -> np.ndarray:
    """Apply thresholding to a coefficient array based on a threshold.
        Data values below threshold are substituted, while those greater
         are shrunk (soft) or left untouched (hard).

    Args:
        coeffs (array_like): array of the wavelet coefficients at a certain level.
        thr_val (float): threshold value for the transformation.
        sub (int, optional): value to substitute the thresholded data with. Defaults to 0.
         mode ({0,1}, optional): select the thresholding mode. Defaults to 1 (soft).

    Returns:
        np.ndarray: resulting thresholded coefficients array.
    """
    # Must check this function!
    if mode == "hard":
        return np.asarray(
            coeffs
            / (np.abs(coeffs) - thr_val)
            * np.maximum(np.abs(coeffs) - thr_val, 0.0)
        )
    else:
        return (
            coeffs
            / np.abs(np.where(np.abs(coeffs) != 0.0, coeffs, 1.0))
            * np.maximum(np.abs(coeffs) - thr_val, 0.0)
        )


@njit
def FirmThreshold(coeffs, val_low, val_high) -> np.ndarray:
    """Apply thresholding to a coefficient array based on a threshold.
        Data values below val_low are thresholded while those above val_high are kept the same.
         Data values within the range are smoothed.

    Args:
        coeffs (array_like): array of the wavelet coefficients at a certain level.
        val_low (float): lower limit for thresholding to 0.
        val_high (float): higher limit for keeping values the same.
    Returns:
        np.ndarray: resulting thresholded coefficients array.
    """
    return np.where(np.abs(coeffs) < val_low, 0.0, coeffs) * np.where(
        np.abs(coeffs) < val_high,
        (
            val_high
            * (1.0 - val_low / np.where(np.abs(coeffs) != 0.0, np.abs(coeffs), 1.0))
            / (val_high - val_low)
        ),
        1.0,
    )

@njit
def LessThreshold(coeffs, thr_val):
    return np.where(np.abs(coeffs)<thr_val,coeffs,0.)



##% Filter implementation for RDataFrame
@ROOT.Numba.Declare(["RVec<double>", "double"], "RVec<double>")
def d_HaarWaveletFilter(signal, threshold):
    # (1) Direct Transform
    # set the scale factor to sqrt(0.5) to preserve l2 norm
    scale = np.sqrt(0.5)
    n = int(np.log2(len(signal)))
    # define the Haar filters (wavelet->high-pass, scaling->low-pass)
    g_filter, h_filter = HaarFunctions()
    # zero-pad the signal up to 2^(n+1) length (if needed)
    t = np.concatenate((signal, np.zeros(np.abs(2 ** (n + 1) - len(signal)))))
    coeff_lst = List()
    for i in range(n):
        # compute the high and low pass coefficients
        # the new low-pass coefficient is processed in the next cycle
        d = np.asarray(nb_convolve(np.flip(t), g_filter)[-2::-2]) * scale
        t = np.asarray(nb_convolve(np.flip(t), h_filter)[-2::-2]) * scale
        # store the high-pass (detail) coefficient, taking out the rightmost 0s
        # by appending up to the last non-0 (argwhere) element with np.argwhere
        coeff_lst.append(d)
    # append the low-pass (approximate) coefficient of the n-th layer
    coeff_lst.append(t)
    coeff_lst = coeff_lst[::-1]
    # there seems to be some issue here with coefficients being set to 0
    # (2) thresholding
    thr_lst = List()
    [thr_lst.append(Threshold(c, threshold, mode="soft")) for c in coeff_lst]
    # (3) Inverse transform
    a, d_coeffs = thr_lst[0], thr_lst[1:]
    inv_h, inv_g = HaarFunctions()
    for dc in d_coeffs:
        # up-sample the coefficients arrays A and D
        up_A = np.zeros(2 * len(a))
        up_A[::2] = a
        up_D = np.zeros(2 * len(dc))
        up_D[::2] = dc
        # pad the up_D array with 0s for the remaining indices
        up_D = np.concatenate((up_D, np.zeros(np.abs(len(up_A) - len(up_D)))))
        # # convolve and add, discarding the last entry
        a = (nb_convolve(up_A, inv_g)[:-1] + nb_convolve(up_D, inv_h)[:-1]) * scale

    return np.asarray(a[: len(signal)])


# Filter with a Firm Threshold
@ROOT.Numba.Declare(["RVec<double>", "double", "double"], "RVec<double>")
def d_HaarWaveletFirm(signal, val_low, val_high):
    # (1) Direct Transform
    # set the scale factor to sqrt(0.5) to preserve l2 norm
    scale = np.sqrt(0.5)
    n = int(np.log2(len(signal)))
    # define the Haar filters (wavelet->high-pass, scaling->low-pass)
    g_filter, h_filter = HaarFunctions()
    # zero-pad the signal up to 2^(n+1) length (if needed)
    t = np.concatenate((signal, np.zeros(np.abs(2 ** (n + 1) - len(signal)))))
    coeff_lst = List()
    for i in range(n):
        # compute the high and low pass coefficients
        # the new low-pass coefficient is processed in the next cycle
        d = np.asarray(nb_convolve(np.flip(t), g_filter)[-2::-2]) * scale
        t = np.asarray(nb_convolve(np.flip(t), h_filter)[-2::-2]) * scale
        # store the high-pass (detail) coefficient, taking out the rightmost 0s
        # by appending up to the last non-0 (argwhere) element with np.argwhere
        coeff_lst.append(d)
    # append the low-pass (approximate) coefficient of the n-th layer
    coeff_lst.append(t)
    coeff_lst = coeff_lst[::-1]
    
    # there seems to be some issue here with coefficients being set to 0
    # (2) thresholding
    thr_lst = List()
    [thr_lst.append(FirmThreshold(c, val_low, val_high)) for c in coeff_lst]
    # test different smoothings for the threshold (like one that depends on the freq.)
    thr_lst[-3] = LessThreshold(thr_lst[-3], 1e-2)
    thr_lst[-4] = LessThreshold(thr_lst[-4], 1e-2)
    # (3) Inverse transform
    a, d_coeffs = thr_lst[0], thr_lst[1:]
    inv_h, inv_g = HaarFunctions()
    for dc in d_coeffs:
        # up-sample the coefficients arrays A and D
        up_A = np.zeros(2 * len(a))
        up_A[::2] = a
        up_D = np.zeros(2 * len(dc))
        up_D[::2] = dc
        # pad the up_D array with 0s for the remaining indices
        up_D = np.concatenate((up_D, np.zeros(np.abs(len(up_A) - len(up_D)))))
        # # convolve and add, discarding the last entry
        a = (nb_convolve(up_A, inv_g)[:-1] + nb_convolve(up_D, inv_h)[:-1]) * scale

    return np.asarray(a[: len(signal)])


##% MAIN for testing
if __name__ == "__main__":
    print("> Define a test waveform")

    t = np.linspace(0, 1, 1000, endpoint=False)
    clean_signal = np.sin(2.0 * np.pi * 10 * t)
    noise = np.random.normal(0, 0.5, clean_signal.shape)
    # waveform = clean_signal + noise
    waveform = np.load("beat.npy")
    thr = 1e-2
    # reconstruction with the custom function
    reco_custom = d_HaarWaveletFirm(waveform,1e-2,1e0)
    # # reconstruction with the pywt library
    # coeff = pywt.wavedec(waveform, "haar", mode="zero")
    # c_thr = [pywt.threshold(c, thr, "soft") for c in coeff]
    # reco_lib = pywt.waverec(c_thr, "haar", mode="zero")

    plt.figure(figsize=(12, 8), facecolor="white")
    plt.title(
        "Soft WaveleT Filter for $thr>0.5$",
        fontsize=18,
    )
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Amp", fontsize=16)
    plt.plot(waveform, label="Noisy signal", linewidth=3)
    plt.plot(reco_custom, label="Custom soft de-noising", linewidth=2)
    #plt.plot(reco_lib, label="Library soft de-noising", linewidth=1)
    plt.tight_layout()
    plt.legend()
    # plt.savefig("./plots/wf_checks/WaveletTFilter_check.pdf")
    plt.show()
