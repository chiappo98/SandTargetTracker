from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
from hitPreprocessing import TimeDistance

def compute_distance3D(Wpoint, WangleX, Tpoint, TtanX, TtanY):
    """compute distance using track and wire equations
    Args:
        Wpoint: point of the wire, coordinates [x,y,z]
        WangleX: wire angle (deg)
        Tpoint: point of the track, coordinates [x,y,z]
        TtanX: tangent in plane XZ
        TtanY: tangent in plane YZ
    """
    Wvec = np.transpose(np.array([np.cos(WangleX*np.pi/180), np.sin(WangleX*np.pi/180), 0]).reshape(3,1) *np.ones((1, TtanX.size)) )
    Tvec = np.transpose(np.array([TtanX, TtanY, np.ones(TtanX.size)]))
    Wpoint = Wpoint.reshape(1,3)
    Tpoint = np.transpose(Tpoint)
    distance = np.abs(np.diag(np.matmul(np.cross(Wvec, Tvec),np.transpose(Tpoint-Wpoint))))
    return distance

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
    Channel = channel+"_c"   # TEMPORARY -> MODIFY DICT
    Z = measureDict[layer][Channel]["height"]
    td.projectToWplane(Z)
    theta = measureDict[layer][Channel]["theta"]
    centroid = measureDict[layer][Channel]["centroid"]
    Tx, Ty = td.projectedX, td.projectedY
    Wpoint = np.array([centroid[0], centroid[1], 0])
    Tpoint = np.array([Tx, Ty, np.zeros(Ty.size)])
    # NOTE: I assume z=0 for points of both wire and tracks
    tanX, tanY = np.array(td.trackList[3]), np.array(td.trackList[4])
    distance = compute_distance3D(Wpoint, theta, Tpoint, tanX, tanY)
    time = (np.array(td.TimeList[0])-np.array(td.TimeList[-1]))
    if plot:
        maxDist = np.sqrt(0.5**2+1)
        fig0, ax0 = plt.subplots(1,2)
        ax0[0].plot(Tx, Ty, '.', markersize=2)
        ax0[1].plot(distance[time>0], time[time>0], '.', markersize=2)
        ax0[1].axvline(np.sqrt(2+1), color='g', linestyle='--')
        fig1, ax1 = plt.subplots(1,2)
        ax1[0].hist(distance,100)
        ax1[0].axvline(maxDist, color='r', linestyle='--')
        ax1[1].hist(time,70)
        plt.figure()
        plt.plot(Tx[distance>maxDist], Ty[distance>maxDist], '.', markersize=2)
        plt.plot(Tx[distance<2.5], Ty[distance<2.5], '.', markersize=2)
        plt.plot(Tx[distance<maxDist], Ty[distance<maxDist], '.', markersize=2)
        plt.figure()
        plt.hist2d(distance[np.logical_and(time>0, distance<2.5)], time[np.logical_and(time>0, distance<2.5)], 70)
        plt.colorbar()
        plt.show()
    return time, distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rootFile', type=str, help='.root file with hits')
    parser.add_argument('measureFile', type=str, help='.json file with computed wire coordinates')
    args = parser.parse_args()
    
    channel = [["dc0", "dc3", "dc0_1", "dc3_1"], ["dc1", "dc4", "dc1_1", "dc4_1"], ["dc2", "dc5", "dc2_1", "dc5_1"]] 
    layer = ["layer_1","layer_2","layer_3"]
    with open(args.measureFile) as f:
        measureDict = json.load(f)
    t_arr, d_arr = np.empty(0), np.empty(0)
    for l in range(0,3):
        for ch in channel[l]:
            t,d = findTime_Distance(args.rootFile, measureDict, ch, layer[l], plot=True)
            t_arr = np.append(t_arr, t)
            d_arr = np.append(d_arr, d)
    plot = True
    if plot:
        plt.hist2d(d_arr[np.logical_and(t_arr>0, d_arr<2.5)], t_arr[np.logical_and(t_arr>0, d_arr<2.5)], 100)
        plt.colorbar()
        plt.xlabel('distance (cm)')
        plt.ylabel('time (s)')
        plt.show()