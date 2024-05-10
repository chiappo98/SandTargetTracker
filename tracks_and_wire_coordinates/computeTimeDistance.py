from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
from scipy.optimize import curve_fit
from scipy.stats import norm
from hitPreprocessing import TimeDistance, Waveforms


def findTime_Distance(trackFile, measureDict, channel, layer, tan_thr, time_delay, plot=False):
    """plot time-distance relation
    Args:
        trackFile: root file with tracks from tracker
        timeFile: root file with time stamps
        measureDict: dictionary with computed wire parameters
        channel: digitizer channel
        layer: wire layer
    """
    td = TimeDistance(trackFile, channel, tan_thr) 
    td.import2RDF()
    Z = measureDict[layer][channel+"_c"]["height"][0]
    theta = measureDict[layer][channel+"_c"]["theta"]
    centroid = measureDict[layer][channel+"_c"]["centroid"]
    Wpoint = np.array([centroid[0], centroid[1], 0])
    td.projectToWplane(Z)
    td.compute_distance3D(Wpoint, theta)
    td.time_calibration(time_delay)
    time = td.DeltaT

    filter_d_tan = (td.distance > 0.1) & (td.ty>0.05) & (td.ty<-0.05)
    
    if plot:
        maxDist = np.sqrt(0.5**2+1)
        fig0, ax0 = plt.subplots(1,2)
        ax0[0].plot(td.newX, td.newY, '.', markersize=2)
        ax0[1].plot(td.distance[time>0], time[time>0], '.', markersize=2)
        ax0[1].axvline(np.sqrt(2+1), color='g', linestyle='--')
        plt.title(channel+" - td.distance")
        fig1, ax1 = plt.subplots(1,2)
        ax1[0].hist(td.distance,100)
        ax1[0].axvline(maxDist, color='r', linestyle='--')
        ax1[1].hist(time,70)
        plt.title(channel+" - time")
        plt.figure()
        plt.plot(td.newX[td.distance>maxDist], td.newY[td.distance>maxDist], '.', markersize=2)
        plt.plot(td.newX[td.distance<2.5], td.newY[td.distance<2.5], '.', markersize=2)
        plt.plot(td.newX[td.distance<maxDist], td.newY[td.distance<maxDist], '.', markersize=2)
        plt.title(channel)
        plt.xlabel("distance (cm)")
        plt.ylabel("time (s)")
        plt.figure()
        plt.hist2d(td.distance[np.logical_and(time>0, td.distance<2.5)], time[np.logical_and(time>0, td.distance<2.5)], 70)
        plt.colorbar()
        plt.title(channel)
        plt.xlabel("distance (cm)")
        plt.ylabel("time (s)")
        plt.show()
    return time, td.distance, td.entryList #[cell_filter]

def plot_sliced_distribution(split_td):
    fig, ax = plt.subplots(5,2)
    i = 0
    for l in split_td:
        plt.figure("time histo")
        plt.hist(l[0,:], 7)
        ax[(i-(i//10)*10)//2,(i//10)].hist(l[1,:], 20 ,histtype='step')
        i += 1
    plt.show()

def fit_td(split_td, func, fit_type, plot=False):
    
    match fit_type:
        case "v_line":
            # compute median
            mean_td = np.array([np.median(l, axis=1) for l in split_td])
            fit_range = (np.abs(mean_td[:,0])>0.2) & (np.abs(mean_td[:,0])<0.5)
        case "pol":
            # fit sliced data with gaussian curve to find distance
            def Gauss(x, a, x0, sigma): 
                return a*np.exp(-(x-x0)**2/(2*sigma**2))
            
            mean_td = []
            for l in split_td:
                h = plt.figure()
                n, bins, patches = plt.hist(l[1,:])
                ppar, pcov = curve_fit(Gauss, bins[:-1] + (bins[1]-bins[0])/2, n , p0=[10, 0, 1], bounds=(0, [np.inf, 1.3, np.inf]), maxfev=1e5)
                plt.close(h)
                mu, std = norm.fit(l[1,:]) 
                mean_td.append(np.array([np.mean(l[0,:]), ppar[1]]))
                
            mean_td = np.array(mean_td)
            fit_range = (mean_td[:,0]>0) & (mean_td[:,0]<1.3e-7) & (mean_td[:,1]>0)
            
    ppar, pcov = curve_fit(func, mean_td[:,0][fit_range], mean_td[:,1][fit_range])
    return ppar, mean_td
    
def plot_td_fit(var1, var2, func, fit_type, plot = False):
    sort_idx = np.argsort(var1)
    split_steps = []
    step = (np.max(var1) - np.min(var1))/50   #10ns bin
    
    for v in np.arange(np.min(var1)+step, np.max(var1), step):
        split_steps.append(np.where(var1[sort_idx]<v)[0][-1])
    split_td = np.array_split(np.vstack((var1[sort_idx],var2[sort_idx])), np.array(split_steps), axis=1)
    
    if plot:
        plot_sliced_distribution(split_td)
        
    fitPar, mean = fit_td(split_td, func, fit_type, plot)
    return fitPar, mean

def display_td_curve(d52, d51, t52, t51, fit_type):
    
    def v_line(x, m, q1, q2):
        y = np.piecewise(x, [x < 0, x >= 0],
                            [lambda x: -m*x + q1, lambda x: m*x + q2])
        return y
    def polinomial(x, m):
        y = m*x
        return y
    
    fig, ax = plt.subplots(2,2)
    
    match fit_type:
        case "v_line":
            ax[0,0].plot(d52[0], t52[0], 'k.', markersize=0.5)
            ax[0,1].plot(d52[1], t52[1], 'k.', markersize=0.5)
            ax[1,0].plot(d51[0], t51[0], 'k.', markersize=0.5)
            ax[1,1].plot(d51[1], t51[1], 'k.', markersize=0.5)
            fitPar52_1, mean52_1 = plot_td_fit(d52[0], t52[0], v_line, fit_type, plot=False)
            fitPar52_2, mean52_2 = plot_td_fit(d52[1], t52[1], v_line, fit_type, plot=False)
            fitPar51_1, mean51_1 = plot_td_fit(d51[0], t51[0], v_line, fit_type, plot=False)
            fitPar51_2, mean51_2 = plot_td_fit(d51[1], t51[1], v_line, fit_type, plot=False)
            ax[0,0].plot(mean52_1[:,0], mean52_1[:,1], 'r.', markersize=3)
            ax[0,1].plot(mean52_2[:,0], mean52_2[:,1], 'r.', markersize=3)
            ax[1,0].plot(mean51_1[:,0], mean51_1[:,1], 'r.', markersize=3)
            ax[1,1].plot(mean51_2[:,0], mean51_2[:,1], 'r.', markersize=3)
            ax[0,0].plot(mean52_1[:,0], v_line(mean52_1[:,0], fitPar52_1[0], fitPar52_1[1], fitPar52_1[2]), 'g--', label=r'v=%.2F $\mu$m/ns' % (1e-5/fitPar52_1[0]))
            ax[0,1].plot(mean52_2[:,0], v_line(mean52_2[:,0], fitPar52_2[0], fitPar52_2[1], fitPar52_2[2]), 'g--', label=r'v=%.2F $\mu$m/ns' % (1e-5/fitPar52_2[0]))
            ax[1,0].plot(mean51_1[:,0], v_line(mean51_1[:,0], fitPar51_1[0], fitPar51_1[1], fitPar51_1[2]), 'g--', label=r'v=%.2F $\mu$m/ns' % (1e-5/fitPar51_1[0]))
            ax[1,1].plot(mean51_2[:,0], v_line(mean51_2[:,0], fitPar51_2[0], fitPar51_2[1], fitPar51_2[2]), 'g--', label=r'v=%.2F $\mu$m/ns' % (1e-5/fitPar51_2[0]))
            ax[0,0].set_xlabel('distance (cm)')
            ax[1,0].set_xlabel('distance (cm)')
            ax[0,1].set_xlabel('distance (cm)')
            ax[1,1].set_xlabel('distance (cm)')
            ax[0,0].set_ylabel('time (s)')
            ax[0,1].set_ylabel('time (s)')
            ax[1,0].set_ylabel('time (s)')
            ax[1,1].set_ylabel('time (s)')
        case "pol":
            # ax[0,0].plot(t52[0], d52[0], 'k.', markersize=0.5)
            # ax[0,1].plot(t52[1], d52[1], 'k.', markersize=0.5)
            # ax[1,0].plot(t51[0], d51[0], 'k.', markersize=0.5)
            # ax[1,1].plot(t51[1], d51[1], 'k.', markersize=0.5)
            ax00 = ax[0,0].hist2d(t52[0], d52[0], bins=[50, 20],cmin = 0)
            ax01 = ax[0,1].hist2d(t52[1], d52[1], bins=[50, 20],cmin = 0)
            ax10 = ax[1,0].hist2d(t51[0], d51[0], bins=[50, 20],cmin = 0)
            ax11 = ax[1,1].hist2d(t51[1], d51[1], bins=[50, 20],cmin = 0)
            # print(ax00[0])
            fig.colorbar(ax00[3], ax=ax[0,0])
            fig.colorbar(ax01[3], ax=ax[0,1])
            fig.colorbar(ax10[3], ax=ax[1,0])
            fig.colorbar(ax11[3], ax=ax[1,1])
            fitPar52_1, mean52_1 = plot_td_fit(t52[0], d52[0], polinomial, fit_type, plot=False)
            fitPar52_2, mean52_2 = plot_td_fit(t52[1], d52[1], polinomial, fit_type, plot=False)
            fitPar51_1, mean51_1 = plot_td_fit(t51[0], d51[0], polinomial, fit_type, plot=False)
            fitPar51_2, mean51_2 = plot_td_fit(t51[1], d51[1], polinomial, fit_type, plot=False)
            ax[0,0].plot(mean52_1[:,0], mean52_1[:,1], 'r.', markersize=5)
            ax[0,1].plot(mean52_2[:,0], mean52_2[:,1], 'r.', markersize=5)
            ax[1,0].plot(mean51_1[:,0], mean51_1[:,1], 'r.', markersize=5)
            ax[1,1].plot(mean51_2[:,0], mean51_2[:,1], 'r.', markersize=5)
            ax[0,0].plot(mean52_1[:,0], polinomial(mean52_1[:,0], fitPar52_1[0]), 'g--', label=r'v=%.2F $\mu$m/ns' % (fitPar52_1[0]*1e-5))
            ax[0,1].plot(mean52_2[:,0], polinomial(mean52_2[:,0], fitPar52_2[0]), 'g--', label=r'v=%.2F $\mu$m/ns' % (fitPar52_2[0]*1e-5))
            ax[1,0].plot(mean51_1[:,0], polinomial(mean51_1[:,0], fitPar51_1[0]), 'g--', label=r'v=%.2F $\mu$m/ns' % (fitPar51_1[0]*1e-5))
            ax[1,1].plot(mean51_2[:,0], polinomial(mean51_2[:,0], fitPar51_2[0]), 'g--', label=r'v=%.2F $\mu$m/ns' % (fitPar51_2[0]*1e-5))
            ax[0,0].set_ylabel('distance (cm)')
            ax[1,0].set_ylabel('distance (cm)')
            ax[0,1].set_ylabel('distance (cm)')
            ax[1,1].set_ylabel('distance (cm)')
            ax[0,0].set_xlabel('time (s)')
            ax[0,1].set_xlabel('time (s)')
            ax[1,0].set_xlabel('time (s)')
            ax[1,1].set_xlabel('time (s)')

    ax[0,0].legend(title=channel[l][0])
    ax[0,1].legend(title=channel[l][1])
    ax[1,0].legend(title=channel[l][2])
    ax[1,1].legend(title=channel[l][3])
    
    plt.show()


def plotWF(channel, entries):
    w = Waveforms(channel) 
    w.import2RDF(350,659, "plot_WFentry", entries[:10])
    w.import2RDF(350,659, "plot_PMTentry", entries[:10])
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('trackFile', type=str, help='.root file with tracks')
    #parser.add_argument('timeFile', type=str, help='.root file with time stamps')
    parser.add_argument('measureFile', type=str, help='.json file with computed wire coordinates')
    parser.add_argument('method', type=str, choices=["all_channels", "dc0", "dc3", "dc0_1", "dc3_1", "dc1", "dc4", "dc1_1", "dc4_1", "dc2", "dc5", "dc2_1", "dc5_1"])
    parser.add_argument('--inspectWF', type=float, nargs='+', help='drift distance range in which inspect waveforms', default=np.nan)
    args = parser.parse_args()
    
    channel = [["dc0", "dc3", "dc0_1", "dc3_1"], ["dc1", "dc4", "dc1_1", "dc4_1"], ["dc2", "dc5", "dc2_1", "dc5_1"]] 
    layer = ["layer_1","layer_2","layer_3"]
    
    with open('../'+args.measureFile) as f:
        measureDict = json.load(f)
        
    if args.method == "all_channels":
        for l in range(0,3):
            t51, d51, t52, d52 = [], [], [], []
            
            for ch in channel[l][:2]:
                t,d, entry = findTime_Distance(args.trackFile, measureDict, ch, layer[l], tan_thr=[-0.1, 0.13], time_delay=0, plot=False) #[-0.05, 0.06]
                filt = (t>0) & (t<0.5e-6) #& (d<1.3)
                t52.append(t[filt])
                d52.append(d[filt])

            for ch in channel[l][2:]:
                t,d, entry = findTime_Distance(args.trackFile, measureDict, ch, layer[l], tan_thr=[-0.13, 0.1], time_delay=50e-9, plot=False) #[-0.06, 0.05]
                filt = (t>0) & (t<0.5e-6) #& (d<1.3)
                t51.append(t[filt])
                d51.append(d[filt])
                 
            display_td_curve(d52, d51, t52, t51, fit_type="pol")

    # following part is not up to date
    if args.method != "all_channels":
        idx = np.where(np.asarray(channel) == args.method)[0][0]
        t,d, entry = findTime_Distance(args.rootFile, measureDict, args.method, layer[idx], plot=False)
        d_filt, t_filt = d[np.logical_and(t>0, d<2.5)], t[np.logical_and(t>0, d<2.5)]
        fitPar = plot_td_fit(d_filt, t_filt, plot=True)
        if args.inspectWF != 0:
            selected_entries = np.array(entry)[np.logical_and(t>0, t>args.inspectWF[0], t<args.inspectWF[1])]
            plotWF(args.method, selected_entries)