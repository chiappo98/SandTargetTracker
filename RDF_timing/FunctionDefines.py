# Where the functions and filters written for the analysis dataframes are defined
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from numba import njit

##% Numba defined functions

@njit(fastmath=True)
def nb_convolve(a, b) -> np.ndarray:
    return np.asarray(np.convolve(a, b))

# Get the Max of a double array
@ROOT.Numba.Declare(["RVec<double>"], "double")
def d_GetWfMax(wf_vec):
    return np.max(wf_vec)


@ROOT.Numba.Declare(["RVec<double>"], "double")
def d_GetDeltaMm(wf_vec):
    return np.max(wf_vec) - np.min(wf_vec)


# Get the M-m difference
@ROOT.Numba.Declare(["RVec<double>"], "double")
def d_GetMmDiff(in_vec):
    return np.abs(np.max(in_vec) - np.min(in_vec))


# Compute the baseline of a waveform/vector and subtract it
@ROOT.Numba.Declare(["RVec<double>", "int"], "RVec<double>")
def d_SubtractBaseline(wf_vec, up_lim):
    return np.asarray(wf_vec - np.mean(wf_vec[:up_lim]))


# Compute the sliding discrete integral of a vector over a window
@ROOT.Numba.Declare(["RVec<double>", "int"], "RVec<double>")
def d_WfSlideInt(wf_vec, window_size):
    slide_int = nb_convolve(wf_vec, np.ones(window_size))
    slide_int[-(window_size - 1):window_size - 1]=0
    return slide_int


# Get the RMS of a waveform/vector
@ROOT.Numba.Declare(["RVec<double>"], "double")
def d_GetRMS(wf_vec):
    return np.sqrt(np.mean(wf_vec**2))


# Get difference between the sliding window integrals of the positive and negative parts
@ROOT.Numba.Declare(["RVec<double>", "int"], "RVec<double>")
def d_GetPNSlideInt(wf_vec, win_size):
    p_wf, n_wf = wf_vec, wf_vec
    p_wf = np.where(p_wf < 0, 0, p_wf)
    n_wf = np.where(n_wf > 0, 0, n_wf)
    return (
        np.convolve(p_wf, np.ones(win_size))[win_size - 1 : -(win_size - 1)]
        + np.convolve(n_wf, np.ones(win_size))[win_size - 1 : -(win_size - 1)]
    )


# Get total scaled integral
@ROOT.Numba.Declare(["RVec<double>", "double"], "double")
def d_GetTotInt(wf_vec, scaling_factor):
    return np.sum(wf_vec) * scaling_factor


@njit
def ComputeT10p(wf_vec):
    val_p = 0.1 * np.max(wf_vec)
    arg_wf_max = np.argmax(wf_vec)
    arg_below = np.argwhere(wf_vec[:arg_wf_max] < val_p)[-1, 0]
    # compute t_10% with a simple interpolation
    if wf_vec[arg_below + 1] > wf_vec[arg_below]:
        return (
            (val_p - wf_vec[arg_below]) / (wf_vec[arg_below + 1] - wf_vec[arg_below])
        ) + arg_below
    else:
        return arg_below


# Get the t_10% index of the chamber waveform
@ROOT.Numba.Declare(["RVec<double>"], "double")
def d_GetChamberT10p(wf_vec):
    return ComputeT10p(wf_vec)


# Get the t_10% index of the chamber waveform
@ROOT.Numba.Declare(["RVec<double>"], "double")
def d_GetPMTT10p(wf_vec):
    wf_vec = -1 * wf_vec
    return ComputeT10p(wf_vec)


# Get the {x,y} projection at height z_plane given the track parameters
@ROOT.Numba.Declare(["double", "double", "double"], "double")
def d_GetXYProjection(r0, s0, delta_z):
    return r0 + s0 * delta_z


@ROOT.Numba.Declare(["double"] * 9, "double")
def d_TrackWireDist(r0_x, r0_y, sx, sy, w0_x, w0_y, w0_z, we_x, we_y):
    # compute the track unit-vector
    e_track = np.array(
        [
            sx,
            sy,
            1.0,
        ]
    ) / np.sqrt(sx**2 + sy**2 + 1)
    # compute the track offset point
    r_track = np.array([r0_x, r0_y, 0.0])

    # get the wire unit and coordinate vectors
    e_wire = np.array([we_x, we_y, 0.0])
    r_wire = np.array([w0_x, w0_y, w0_z])
    # compute the wire-track distance
    v_norm = np.cross(e_wire, e_track)
    # print(np.linalg.norm(v_norm))
    # tw_dist = np.abs(np.dot(v_norm, (r_wire - r_track))) / np.linalg.norm(v_norm)
    return np.abs(np.dot(v_norm, (r_wire - r_track))) / np.linalg.norm(
        v_norm
    )  # tw_dist

@ROOT.Numba.Declare(["double"] * 2, "double")
def d_PolarAngle(sx,sy):
    return np.arccos(1/np.sqrt(sx**2+sy**2+1))

##% Numba defined Filters


# Select events with some quantity above a threshold
@ROOT.Numba.Declare(["double", "float"], "bool")
def f_ValAboveThr(value, threshold):
    return value > threshold


# Select events with some quantity below a threshold
@ROOT.Numba.Declare(["double", "float"], "bool")
def f_ValBelowThr(value, threshold):
    return value < threshold


# Select events with some quantity below a threshold
@ROOT.Numba.Declare(["double", "float", "float"], "bool")
def f_ValInRange(value, v_min, v_max):
    return (bool)(value > v_min and value < v_max)


# Select events with waveform M-m below a threshold
@ROOT.Numba.Declare(["RVec<double>", "double"], "bool")
def f_DeltaBelowVal(in_vec, delta):
    return np.abs(np.max(in_vec) - np.min(in_vec)) < delta


# Select events with waveform M-m above a threshold
@ROOT.Numba.Declare(["RVec<double>", "double"], "bool")
def f_DeltaAboveVal(in_vec, delta):
    return np.abs(np.max(in_vec) - np.min(in_vec)) > delta


# Select events with waveform M-m within a range
@ROOT.Numba.Declare(["RVec<double>", "double", "double"], "bool")
def f_DeltaInRange(in_vec, int_min, int_max):
    range = np.abs(np.max(in_vec) - np.min(in_vec))
    return (bool)(range > int_min and range < int_max)


# Select events with Max within a range
@ROOT.Numba.Declare(["RVec<double>", "double", "double"], "bool")
def f_MaxInRange(in_vec, int_min, int_max):
    range = np.abs(np.max(in_vec))
    return (bool)(range > int_min and range < int_max)


# Select events with Max below some value
@ROOT.Numba.Declare(["RVec<double>", "double"], "bool")
def f_MinBelowVal(in_vec, val):
    return np.min(in_vec) < val


# Select events with FFT within a certain range up to some frequency
@ROOT.Numba.Declare(["RVec<double>", "int", "double", "double"], "bool")
def f_FFTInRange(FFT_vec, max_idx, fft_min, fft_max):
    return (bool)(
        np.min(FFT_vec[:max_idx]) > fft_min and np.max(FFT_vec[:max_idx]) < fft_max
    )


##% C++ defined functions: a function returns the string to be interpreted


def GetCppCode():
    return """
    ROOT::VecOps::RVec<double> d_ComputeFFT(const ROOT::VecOps::RVec<double> &wf){
        // get the number of waveform points
        int wf_dim = wf.size();
        // set up the FFT
        std::unique_ptr<TVirtualFFT> fft(TVirtualFFT::FFT(1,&wf_dim,"R2C"));
        fft->SetPoints(wf.data());
        // apply the transform
        fft->Transform();
        // define Re and Im components
        double re = 0, im = 0;
        // define the output vector
        ROOT::VecOps::RVec<double> fft_out;
        // extract the fft magnitude
        for(int idx = 0; idx < (wf_dim/2); idx++){
            fft->GetPointComplex(idx, re, im);
            fft_out.push_back(TMath::Sqrt((re*re+im*im)/wf_dim));
        }
    return fft_out;
    }
    
    ROOT::VecOps::RVec<double> d_FilteredInverseFFT(const ROOT::VecOps::RVec<double> &wf, const double max_freq){
        // get the number of waveform points
        int wf_dim = wf.size();
        // sample spacing 
        const double sample_spc = 1e-9;
        // set up the FFT
        TVirtualFFT *fft(TVirtualFFT::FFT(1,&wf_dim,"R2C"));
        fft->SetPoints(wf.data());
        // apply the transform
        fft->Transform();
        // define vectors for the real and im. components
        ROOT::VecOps::RVec<double> re_vec;
        ROOT::VecOps::RVec<double> im_vec;
        // define Re and Im components
        double re = 0, im = 0;
        // define pointers for storing the Re and Im parts
        //double *re_full = new double[wf_dim];
        //double *im_full = new double[wf_dim];
        // fill the component vectors
        //fft->GetPointsComplex(re_full,im_full);
        for(int idx = 0; idx < wf_dim; idx++){
            fft->GetPointComplex(idx, re, im);
            re_vec.push_back(re);
            im_vec.push_back(im);
        }
        //re_vec = std::vector<double>(re_full,re_full+wf_dim);
        //re_vec = std::vector<double>(im_full,im_full+wf_dim);
        
        // compute the index in the frequency series corresponding to the limit
        const int p_idx = max_freq * (sample_spc * wf_dim);
        if(wf_dim % 2 == 0){
            // fill the component vectors with 0
            std::fill(re_vec.begin()+p_idx, re_vec.end()-p_idx+1,0.);
            std::fill(im_vec.begin()+p_idx, im_vec.end()-p_idx+1,0.);
            }
        else {
            // fill the component vectors with 0
            std::fill(re_vec.begin()+p_idx, re_vec.end()-p_idx+1,0.);
            std::fill(im_vec.begin()+p_idx, im_vec.end()-p_idx+1,0.);
        }
        // set up the inverse transform
        TVirtualFFT *i_fft(TVirtualFFT::FFT(1,&wf_dim,"C2R"));
        i_fft->SetPointsComplex(re_vec.data(),im_vec.data());
        // apply the inverse transform
        i_fft->Transform();
        // // reset the re/im vectors
        re_vec.clear();
        im_vec.clear();
        //extract the real part
        //re_full = i_fft->GetPointsReal();
        //re_vec = std::vector<double>(re_full,re_full+wf_dim);
        //std::transform(re_vec.begin(), re_vec.end(), re_vec.begin(), [&wf_dim](auto& c){return c / wf_dim;});
        
        for(int idx = 0; idx < wf_dim; idx++){
            i_fft->GetPointComplex(idx, re, im);
            re_vec.push_back(re / wf_dim);
            //im_vec.push_back(im / wf_dim);
        }
        return re_vec;
    }
    
    ROOT::VecOps::RVec<double> d_FilterFFTFreqs(const ROOT::VecOps::RVec<double> &wf, 
                                            std::vector<double> filter_freqs, 
                                            const double f_range){
        // get the number of waveform points
        int wf_dim = wf.size();
        // sample spacing 
        const double sample_spc = 1e-9;
        // set up the FFT
        TVirtualFFT *fft(TVirtualFFT::FFT(1,&wf_dim,"R2C"));
        fft->SetPoints(wf.data());
        // apply the transform
        fft->Transform();
        // define vectors for the real and im. components
        ROOT::VecOps::RVec<double> re_vec;
        ROOT::VecOps::RVec<double> im_vec;
        // define Re and Im components
        double re = 0, im = 0;
        // fill the component vectors
        //fft->GetPointsComplex(re_vec.data(),im_vec.data());
        for(int idx = 0; idx < wf_dim; idx++){
            fft->GetPointComplex(idx, re, im);
            re_vec.push_back(re);
            im_vec.push_back(im);
        }
        // compute the index in the frequency series corresponding to the limit
        for(auto &freq : filter_freqs){
            const int f_idx = freq * (sample_spc * wf_dim);
            const int range_idx = (f_range/2) * (sample_spc * wf_dim);
            if(wf_dim % 2 == 0){
                // notch-down the positive frequencies
                std::fill(re_vec.begin()+f_idx-range_idx, re_vec.begin()+f_idx+range_idx,0.);
                std::fill(re_vec.begin()+f_idx-range_idx, re_vec.begin()+f_idx+range_idx,0.);
                // notch-down the negative frequencies
                std::fill(re_vec.end()-f_idx-range_idx, re_vec.end()-f_idx+range_idx,0.);
                std::fill(im_vec.end()-f_idx-range_idx, im_vec.end()-f_idx+range_idx,0.);
                }
            else {
                // notch-down the positive frequencies
                std::fill(re_vec.begin()+f_idx-range_idx, re_vec.begin()+f_idx+range_idx,0.);
                std::fill(re_vec.begin()+f_idx-range_idx, re_vec.begin()+f_idx+range_idx,0.);
                // notch-down the negative frequencies
                std::fill(re_vec.end()-f_idx-range_idx, re_vec.end()-f_idx+range_idx,0.);
                std::fill(im_vec.end()-f_idx-range_idx, im_vec.end()-f_idx+range_idx,0.);
            }
        }
        // set up the inverse transform
        TVirtualFFT *i_fft(TVirtualFFT::FFT(1,&wf_dim,"C2R"));
        i_fft->SetPointsComplex(re_vec.data(),im_vec.data());
        // apply the inverse transform
        i_fft->Transform();
        // reset the re/im vectors
        re_vec.clear();
        im_vec.clear();
        //extract the real part
        for(int idx = 0; idx < wf_dim; idx++){
            i_fft->GetPointComplex(idx, re, im);
            re_vec.push_back(re / wf_dim);
            //im_vec.push_back(im / wf_dim);
        }
        
        return re_vec;
    }
    
    ROOT::RDF::RResultPtr<ULong64_t> AddProgressBar(ROOT::RDF::RNode df) {
        auto c = df.Count();
        c.OnPartialResult(/*every=*/10, [] (ULong64_t e) { std::cout << e << std::endl; });
        return c;
    }
    
    """


##% MAIN for testing
if __name__ == "__main__":
    ROOT.gInterpreter.Declare(GetCppCode())

    time_srs = np.arange(0, 200, 0.1)
    sine_wave1 = np.sin(time_srs)
    sine_wave2 = np.sin(0.5 * time_srs)
    np_sine_wave = sine_wave1 + sine_wave2
    plt.figure(figsize=(10, 8))
    plt.plot(time_srs, np_sine_wave)
    plt.show()

    np_fft = np.asarray(ROOT.d_ComputeFFT(np_sine_wave))
    plt.plot(time_srs[:1000] / (200 * 0.1), np_fft)
    plt.show()
    wave_freq = np.argmax(np_fft) / (200 * 0.1)
    print(wave_freq)
    np_filtered = np.asarray(ROOT.d_FilteredInverseFFT(np_sine_wave, 0.12))
    plt.figure(figsize=(10, 8))
    plt.tight_layout()
    # plt.plot(time_srs, sine_wave1,label="sine_wave1")
    plt.plot(time_srs, sine_wave2, label="sine_wave2")
    # plt.plot(time_srs,np_sine_wave,label="sum_wave")
    plt.plot(time_srs, np_filtered, label="iffT")
    plt.legend()
    plt.show()
