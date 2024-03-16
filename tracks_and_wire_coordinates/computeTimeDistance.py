from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
from scipy.optimize import curve_fit
from hitPreprocessing import TimeDistance, Waveforms, Distance


def findTime_Distance(hitFile, measureDict, channel, layer, plot=False):
    """plot time-distance relation
    Args:
        hitFile: root file with hits
        measureDict: dictionary with computed wire parameters
        channel: digitizer channel
        layer: wire layer
    """
    td = TimeDistance(hitFile, channel, 0.02e-9) 
    td.import2RDF()
    Z = measureDict[layer][channel+"_c"]["height"]
    theta = measureDict[layer][channel+"_c"]["theta"]
    centroid = measureDict[layer][channel+"_c"]["centroid"]
    Wpoint = np.array([centroid[0], centroid[1], 0])
    td.projectToWplane(Z)
    td.compute_distance3D(Wpoint, theta)
    time = (np.array(td.TimeList[0])-np.array(td.TimeList[-1]))
    if plot:
        maxDist = np.sqrt(0.5**2+1)
        fig0, ax0 = plt.subplots(1,2)
        ax0[0].plot(td.projectedX, td.projectedY, '.', markersize=2)
        ax0[1].plot(td.distance[time>0], time[time>0], '.', markersize=2)
        ax0[1].axvline(np.sqrt(2+1), color='g', linestyle='--')
        plt.title(channel+" - td.distance")
        fig1, ax1 = plt.subplots(1,2)
        ax1[0].hist(td.distance,100)
        ax1[0].axvline(maxDist, color='r', linestyle='--')
        ax1[1].hist(time,70)
        plt.title(channel+" - time")
        plt.figure()
        plt.plot(td.projectedX[td.distance>maxDist], td.projectedY[td.distance>maxDist], '.', markersize=2)
        plt.plot(td.projectedX[td.distance<2.5], td.projectedY[td.distance<2.5], '.', markersize=2)
        plt.plot(td.projectedX[td.distance<maxDist], td.projectedY[td.distance<maxDist], '.', markersize=2)
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
    return time, td.distance, td.entryList

def plot_sliced_distribution(split_td):
    fig, ax = plt.subplots(5,2)
    i = 0
    for l in split_td:
        plt.figure("distance histo")
        plt.hist(l[0,:], 7)
        ax[(i-(i//10)*10)//2,(i//10)].hist(l[1,:], 20 ,histtype='step')
        i += 1
    plt.show()

def fit_td(split_td, plot=False):
    mean_td = [np.mean(l, axis=1) for l in split_td]
    mean_td = np.array(mean_td)
    def line(x, m, q):
        y = m*x + q
        return y
    ppar, pcov = curve_fit(line, mean_td[:,0][mean_td[:,0]<1], mean_td[:,1][mean_td[:,0]<1])
    if plot:
        fig0 = plt.figure()
        plt.plot(mean_td[:,0], mean_td[:,1], '.')
        plt.plot(mean_td[:,0][mean_td[:,0]<1], line(mean_td[:,0][mean_td[:,0]<1], ppar[0], ppar[1]), '--', label=r'v=%.2F $\mu$m/ns' % (1e-5/ppar[0]))
        plt.xlabel('distance (cm)')
        plt.ylabel('time (s)')
        plt.legend()
        fig1 = plt.figure()
        # plt.plot(d,t,'.', markersize=2)
        plt.hist2d(d, t, 70)
        plt.plot(mean_td[:,0][mean_td[:,0]<1], line(mean_td[:,0][mean_td[:,0]<1], ppar[0], ppar[1]), 'r--', label=r'v=%.2F $\mu$m/ns' % (1e-5/ppar[0]))
        plt.xlabel('distance (cm)')
        plt.ylabel('time (s)')
        plt.legend()
        plt.colorbar()
        plt.show()   
        return 0
    
def plot_td_fit(d, t, plot = False):
    sort_idx = np.argsort(d)
    split_steps = []
    for v in np.arange(2.5/20, 2.5, 2.5/20):
        split_steps.append(np.where(d[sort_idx]<v)[0][-1])
    split_td = np.array_split(np.vstack((d[sort_idx],t[sort_idx])), np.array(split_steps), axis=1)
    plot_sliced_distribution(split_td)
    fit_td(split_td, plot=True)
    return 0

def plotWF(channel, entries):
    w = Waveforms(channel) 
    w.import2RDF(350,659, "plot_WFentry", entries[:10])
    w.import2RDF(350,659, "plot_PMTentry", entries[:10])
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rootFile', type=str, help='.root file with hits')
    parser.add_argument('measureFile', type=str, help='.json file with computed wire coordinates')
    parser.add_argument('method', type=str, choices=["all_channels", "dc0", "dc3", "dc0_1", "dc3_1", "dc1", "dc4", "dc1_1", "dc4_1", "dc2", "dc5", "dc2_1", "dc5_1"])
    parser.add_argument('--inspectWF', type=float, nargs='+', help='drift distance range in which inspect waveforms', default=np.nan)
    args = parser.parse_args()
    
    channel = [["dc0", "dc3", "dc0_1", "dc3_1"], ["dc1", "dc4", "dc1_1", "dc4_1"], ["dc2", "dc5", "dc2_1", "dc5_1"]] 
    layer = ["layer_1","layer_2","layer_3"]
    with open(args.measureFile) as f:
        measureDict = json.load(f)
    if args.method == "all_channels":
        t_arr, d_arr, entry_arr = np.empty(0), np.empty(0), np.empty(0)
        for l in range(0,3):
            for ch in channel[l]:
                t,d, entry = findTime_Distance(args.rootFile, measureDict, ch, layer[l], plot=False)
                t_arr = np.append(t_arr, t)
                d_arr = np.append(d_arr, d)
                entry_arr = np.append(entry_arr, entry)
        plot = False
        if plot:
            plt.hist2d(d_arr[np.logical_and(t_arr>0, d_arr<2.5)], t_arr[np.logical_and(t_arr>0, d_arr<2.5)], 100)
            plt.colorbar()
            plt.xlabel('distance (cm)')
            plt.ylabel('time (s)')
            plt.show()
    if args.method != "all_channels":
        idx = np.where(np.asarray(channel) == args.method)[0][0]
        t,d, entry = findTime_Distance(args.rootFile, measureDict, args.method, layer[idx], plot=False)
        d_filt, t_filt = d[np.logical_and(t>0, d<2.5)], t[np.logical_and(t>0, d<2.5)]
        plot_td_fit(d_filt, t_filt, plot=True)
        if args.inspectWF != 0:
            selected_entries = np.array(entry)[np.logical_and(t>0, t>args.inspectWF[0], t<args.inspectWF[1])]
            plotWF(args.method, selected_entries)