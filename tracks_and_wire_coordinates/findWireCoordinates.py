from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
from scipy.optimize import curve_fit
from hitPreprocessing import TrackPosition

def compute_distance(dot_x, dot_y, line_x, line_y):
    p1 = np.array([line_x[0], line_y[0]])
    p2 = np.array([line_x[-1], line_y[-1]])
    p3 = np.array([dot_x, dot_y])
    distance = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
    return distance

def findXYwire(hitFile, channel, plotSigma=False, plotProfile=False):
    """find best values for : angle and intercept (with x=0 axis) using rotation method
     Args:
        hitFile: root file with hits from tracker
        channel: digitizer channel, and name of TTree Branch
        plot: plot sigma of hit distribution projected on the Y axis wrt rotation angle of the wire, default=False
     Returns:
        wire angle
    """    
    tracks = TrackPosition(hitFile, channel, 0.02e-9, 0.05)
    tracks.import2RDF()
    tracks.WireAngle = np.arange(-12,6,0.2)
    
    for t in tracks.WireAngle:
        tracks.projectToWplane()
        tracks.rotate(t)
        tracks.fit_profile()
        tracks.sigmaAngle = np.append(tracks.sigmaAngle, abs(tracks.pyPar_at_Theta[2]))
        tracks.WirePosition = np.append(tracks.WirePosition, tracks.pyPar_at_Theta[1])
    tracks.fit_poly2d(tracks.WireAngle, 2, plotSigma)
    tracks.rotate(tracks.minAngle)
    tracks.fit_profile()
    tracks.plotTracks("wireFit")
    if plotProfile:
        tracks.rotate(tracks.minAngle)
        tracks.fit_profile(True)    
    
    return tracks.minAngle#, wirePosition

def findXYwire_pca(filename, channel, plotPCA=False, plotProfile=False):
    """find best values for : angle and intercept (with x=0 axis) using PCA method
     Args:
        hitFile: root file with hits from tracker
        channel: digitizer channel, and name of TTree Branch
        plotPCA: plot results from PCA
        plotProfile: plot sigma of hit distribution projected on the Y axis wrt rotation angle of the wire, default=False
     Returns:
        wire angle
    """  
    tracksPCA = TrackPosition(filename, channel, 0.02e-9, 0.05) 
    tracksPCA.import2RDF()
    tracksPCA.projectToWplane()
    
    tracksPCA.pca(plotPCA)

    minimumAngle = tracksPCA.pca_angle
    tracksPCA.rotate(minimumAngle)
    tracksPCA.fit_profile(plotProfile)
    tracksPCA.plotTracks("wireFit")
    plt.legend()  
    
    return minimumAngle, #wirePosition

def findXYZ(hitFile, coordinateFile, channel, plane, plotZSigma=False, plotProfile=False):
    """find best values for : wire height, angle and intercept (with x=0 axis)
     Args:
        hitFile: root file with hits from tracker
        coordinateFile: json file with vertical distances between detector components
        channel: digitizer channel, and name of TTree Branch
        plane: number of the wire plane, from 0 (top), to 2 (bottom)
        plotZSigma: plot sigma of hit distribution projected on the Y axis wrt Z coordinate of the wire, default=False
     Returns:
        height at which hits stored in the TTree are computed
        wire height(wrt TTree height), angle, intercept and sigma
    """
    tracks = TrackPosition(hitFile, channel, 0.02e-9, 0.02) 
    tracks.import2RDF("new_tracks") #charge
    tracks.readZmeasures("../"+coordinateFile)
    hTracker = np.mean(tracks.trackList[2])
    height = np.arange(-6, 12, 0.3) + tracks.hplanes[plane] #-6
    for h in height:
        tracks.projectToWplane(h)
        tracks.pca()
        tracks.rotate(tracks.pca_angle)
        tracks.fit_profile()
        tracks.sigmaAngle = np.append(tracks.sigmaAngle, abs(tracks.pyPar_at_Theta[2]))
        tracks.WireAngle = np.append(tracks.WireAngle, tracks.pca_angle)
    tracks.fit_poly2d(height, 4, plotZSigma)
    tracks.projectToWplane(tracks.minH)
    tracks.pca()
    tracks.rotate(tracks.minAngle)
    tracks.fit_profile()
    minPosition = [tracks.pyPar_at_Theta[1], tracks.perr_at_Theta[1]]
    minCentroid = (tracks.centroid).tolist()
    plothits= False
    if plothits:
        tracks.plotTracks("original")
        tracks.plotTracks("projected")
        tracks.plotTracks("wireFit")
        plt.legend()
        plt.show()
    if plotProfile:
        tracks.rotate(tracks.minAngle)
        tracks.fit_profile(True)
    return hTracker, [tracks.minH, tracks.minH_err, tracks.pca_angle, minPosition, minCentroid, tracks.minSigma]

def line_intersection(x_i, x_f, y_i, y_f):
    """find intersections between two lines
    Args: 
        start/end x/y coordinates of the two wires. Format: list [wire1, wire2]
    Returns: intersection coordinates 
    """
    xdiff = (x_i[0] - x_f[0], x_i[1] - x_f[1])
    ydiff = (y_i[0] - y_f[0], y_i[1] - y_f[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if not div == 0:
        d = (det(*((x_i[0], y_i[0]),(x_f[0], y_f[0]))), det(*((x_i[1], y_i[1]),(x_f[1], y_f[1]))))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
    else:
        x, y = np.nan, np.nan
    return x, y

def findWireIntersections(coord_dict, layer, channel):
    """find intersections between all wires
    Args: 
        coord_dict: dictionary with wire coordinates
        layer: list of layers (to read the dictionary)
        channel: list of channels (to read the dictionary)
    """    
    y0 = []
    theta= []
    for i in range(0,3):
        for ch in channel[i]:
            y0.append(coord_dict[layer[i]][ch]["intercept"][0])
            theta.append(coord_dict[layer[i]][ch]["theta"])
    # maybe the previous part can be included in the creation of the dict, so that we can pass
    # just two lists (y0 and theta) to this func.
    y_i = np.transpose(np.array(y0).reshape(-1,2))        
    theta = np.transpose(np.array(theta).reshape(-1,2))
    x_i, x_f = [0,0], [40,40] 
    y_f = y_i + np.sin(theta*np.pi/180)*x_f[0]
    for i in range(2):
        for j in range(1,3):
            int_x, int_y = line_intersection(x_i, x_f, [y_i[i][0], y_i[i][j]], [y_f[i][0], y_f[i][j]])
            print(int_x,int_y)
            plt.plot([x_i[0], x_f[0]], [y_i[i][0], y_f[i][0]], '--')
            plt.plot([x_i[0], x_f[0]], [y_i[i][j], y_f[i][j]], '--')
            plt.plot(int_x,int_y, 'o')
    plt.ylim([0,40])
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()
    return 0

def compute_geometry(layer):
    #NOT READY
    with open("./prototype_toy_model/geo_config.json") as f:
        geo_dict = json.load(f)
    z_off, y_off, ncells = [], [], []    
    pitch = geo_dict["wire_spacing"]
    theta = np.array([5, 0, -5])
    for l in layer:
        z_off.append(geo_dict[l]["x_off"])
        y_off.append(geo_dict[l]["x_off"])
        ncells.append(geo_dict[l]["n_cells"])
    layer1Y = np.linspace(y_off[0], y_off[0]+ncells[0]*pitch, ncells[0])
    layer2Y = np.linspace(y_off[1], y_off[1]+ncells[1]*pitch, ncells[1])
    layer3Y = np.linspace(y_off[2], y_off[2]+ncells[2]*pitch, ncells[2])   
    
    Ywires_i = np.array([layer1Y[0], layer2Y[1], layer2Y[0]])
    Xwires_i = 0#np.array(x_off)
    Xwires_f = Xwires_i+40
    Ywires_f = Ywires_f + np.sin(theta*np.pi/180)*Xwires_f
    
    return 0

def linearFunc(x,intercept):
    return intercept


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rootFile', type=str, help='.root file with hits')
    parser.add_argument('coordFile', type=str, help='.json file with distances between detector components')
    parser.add_argument('method', type=str, help='what to compute: XYrot, XYpca, XYZpca')
    args = parser.parse_args()
    
    wireYpos = np.empty(0)
    wireAngle = np.empty(0)
    
    match args.method:
        case "XYrot":
            plane = [0,1,2,0,1,2]
            channel= ["dc0_c", "dc1_c", "dc2_c", "dc3_c", "dc4_c", "dc5_c"]
            fig = plt.figure()
            for ch in channel:
                angle = findXYwire("20240226_STT_runs350_442.root", ch)  #args.rootFile
                wireAngle = np.append(wireAngle, angle)
                # wireYpos = np.append(wireYpos, Ypos)
            plt.xlabel('x (cm)')
            plt.ylabel('y (cm)')
            plt.title('')
            plt.legend()
            plt.show()

        case "XYpca":
            plane = [0,1,2,0,1,2]
            channel= ["dc0_c", "dc1_c", "dc2_c", "dc3_c", "dc4_c", "dc5_c"]
            figPCA = plt.figure()
            for ch in channel:
                angle = findXYwire_pca("20240226_STT_runs350_442.root" ,ch)
                wireAngle = np.append(wireAngle, angle)
                # wireYpos = np.append(wireYpos, Ypos)
            plt.title('')
            plt.show()
        
        case "XYZpca":
            channel = [["dc0_c", "dc3_c", "dc0_1_c", "dc3_1_c"], ["dc1_c", "dc4_c", "dc1_1_c", "dc4_1_c"], ["dc2_c", "dc5_c", "dc2_1_c", "dc5_1_c"]] 
            plane = [0,1,2]
            layer = ["layer_1","layer_2","layer_3"]
            coord = ["height", "theta", "intercept", "centroid", "sigmaAngle"]
            output_dict = {}
            wire_id = np.array([1,2,3,4])
            colors = ['r', 'g', 'b']
            fh, hAx = plt.subplots()
            fa, aAx = plt.subplots()
            for l in range(0,3):
                layer_dict = {}
                h, herr, angle, position, posErr= [], [], [], [], []
                print(layer[l])
                for ch in channel[l]:
                    hTracker, values = findXYZ(args.rootFile, args.coordFile, ch, plane[l], plotZSigma=False, plotProfile=False)
                    layer_dict[ch] = dict(zip(coord, values))
                    print(ch, values[0]+hTracker, values[2:])
                    h.append(values[0])
                    herr.append(values[1])
                    angle.append(values[2])
                output_dict[layer[l]] = layer_dict
                print("height:", np.mean(np.array(h)), '+/-',np.std(np.array(h)))
                print("angle:", np.mean(np.array(angle)), '+/-',np.std(np.array(angle)))
                
                # summary plot
                hAx.errorbar(wire_id, h, np.abs(np.array(herr)), fmt="." ,color=colors[l])
                aAx.errorbar(wire_id, angle, [0.3,0.3,0.3,0.3], fmt=".", color=colors[l])   # still don't know how to compute pca errors 
                #fit data points
                h_fit, h_cov=curve_fit(linearFunc,wire_id,h,) #sigma=herr
                a_fit, a_cov=curve_fit(linearFunc,wire_id,angle)  #,sigma=[0.1,0.1,0.1,0.1]
                print(np.sqrt(h_cov[0][0]))
                print(np.sqrt(a_cov[0][0]))
                hAx.axhline(h_fit[0] ,color=colors[l], linestyle='-', label='{:.2f}'.format(h_fit[0])+'+/-'+'{:.2f}'.format(np.sqrt(h_cov[0][0])))
                aAx.axhline(a_fit[0] ,color=colors[l], linestyle='-', label='{:.2f}'.format(a_fit[0])+'+/-'+'{:.2f}'.format(np.sqrt(a_cov[0][0])))
            hAx.set_ylabel("vertical position (cm)")
            aAx.set_ylabel("angle (°)")
            hAx.set_xlabel("wire ID")
            aAx.set_xlabel("wire ID")
            hAx.legend()
            aAx.legend()
            plt.show()
            output_dict["h0_hits"] = hTracker
            
            #findWireIntersections(output_dict, layer, channel)
            
            with open("../measures_20240328.json", "w") as outfile: 
                json.dump(output_dict, outfile)