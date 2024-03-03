from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
from hitPreprocessing import TrackPosition

def compute_distance(dot_x, dot_y, line_x, line_y):
    p1 = np.array([line_x[0], line_y[0]])
    p2 = np.array([line_x[-1], line_y[-1]])
    p3 = np.array([dot_x, dot_y])
    distance = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
    return distance

def findXYwire(filename, channel, plot=False):
    wirePos = np.empty(0)
    sigma = np.empty(0)
    angle = np.arange(-12,6,0.2)
    tracks = TrackPosition(filename, channel, 0.02e-9)
    tracks.import2RDF()
    
    for t in angle:
        tracks.projectToWplane()
        tracks.rotate(t)
        tracks.fit_profile()
        wirePos = np.append(wirePos, tracks.pyPar_at_Theta[1])
        sigma = np.append(sigma, abs(tracks.pyPar_at_Theta[2]))
    sigmaFit = np.polyfit(angle, sigma, 4)
    sigmaCurve = np.poly1d(sigmaFit)
    minimumAngle = angle[np.array(sigmaCurve(angle)) == np.min(np.array(sigmaCurve(angle)))][0]
    tracks.rotate(minimumAngle)
    tracks.fit_profile()
    wirePosition = tracks.pyPar_at_Theta[1]
    xline = np.linspace(0,40,40)*np.cos(minimumAngle* np.pi/180)
    yline = float(wirePosition) + np.linspace(0,40,40)*np.sin(minimumAngle* np.pi/180)
    plt.plot(tracks.trackList[0],tracks.trackList[1], '.', markersize=2)
    plt.plot(xline, yline, '--', label=channel[:3]+r': $\theta$='+ '{:.3f}'.format(minimumAngle)+r'° , $\sigma$='+ '{:.2f}'.format(tracks.pyPar_at_Theta[2]))
    plt.legend()
    if plot:
        tracks.rotate(minimumAngle)
        tracks.fit_profile(True)
        
        fig, ax = plt.subplots(1,2)
        ax[0].plot(angle, wirePos, '.', markersize=2)
        ax[0].axvline(x = angle[sigma==np.min(abs(sigma))][0], color='g', linestyle='--', label='minimum sigma')
        ax[0].axhline(y = wirePos[sigma==np.min(abs(sigma))][0], color='orange', linestyle='--')
        ax[0].set_xlabel('angle (°)')
        ax[0].set_ylabel('wire coordinate')
        
        ax[1].plot(angle, sigma, '.')
        ax[1].plot(angle, sigmaCurve(angle), '--')
        ax[1].set_xlabel('angle (°)')
        ax[1].set_ylabel('sigma')
        fig.legend()
        plt.show()    
    
    return minimumAngle#, wirePosition

def findXYwire_pca(filename, channel, plotPCA=False, plotProfile=False):
    tracksPCA = TrackPosition(filename, channel, 0.02e-9) 
    tracksPCA.import2RDF()
    tracksPCA.projectToWplane()
    
    tracksPCA.pca(plotPCA)

    minimumAngle = tracksPCA.pca_angle
    tracksPCA.rotate(minimumAngle)
    tracksPCA.fit_profile(plotProfile)
    tracksPCA.plotTracks("wireFit")
    plt.legend()  
    
    return minimumAngle, #wirePosition

def findXYZ(hitFile, coordinateFile, channel, plane, plotZSigma=False):
    tracks = TrackPosition(hitFile, channel, 0.02e-9) 
    tracks.import2RDF()
    tracks.readZmeasures(coordinateFile)
    hTracker = np.mean(tracks.trackList[2])
    height = np.arange(-6, 12, 0.3) + tracks.hplanes[plane]
    for h in height:
        tracks.projectToWplane(h)
        tracks.pca()
        tracks.rotate(tracks.pca_angle)
        tracks.fit_profile()
        tracks.sigmaAngle = np.append(tracks.sigmaAngle, abs(tracks.pyPar_at_Theta[2]))
        tracks.WireAngle = np.append(tracks.WireAngle, tracks.pca_angle)
        tracks.WirePosition = np.append(tracks.WirePosition, tracks.pyPar_at_Theta[1])
    tracks.fit_poly2d(height, plotZSigma)
    # tracks.plotTracks("original")
    # tracks.plotTracks("projected")
    # tracks.plotTracks("wireFit")
    # plt.legend()
    # plt.show()
    return hTracker, [tracks.minH, tracks.minAngle, tracks.minPosition, tracks.minSigma]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rootFile', type=str, help='.root file with hits')
    parser.add_argument('coordFile', type=str, help='.json file with distances between detector components')
    args = parser.parse_args()
    
    wireYpos = np.empty(0)
    wireAngle = np.empty(0)
    channel = [["dc0_c", "dc3_c"], ["dc1_c", "dc4_c"], ["dc2_c", "dc5_c"]] 
    plane = [0,1,2]
    layer = ["layer_1","layer_2","layer_3"]
    coord = ["height", "theta", "intercept", "sigmaPosition"]
    # plane = [0,1,2,0,1,2]
    # channel= ["dc0_c", "dc1_c", "dc2_c", "dc3_c", "dc4_c", "dc5_c"]
    # # fig = plt.figure()
    # # for ch in channel:
    # #     angle = findXYwire("20240226_STT_runs350_442.root", ch)  #args.rootFile
    # #     wireAngle = np.append(wireAngle, angle)
    # #     # wireYpos = np.append(wireYpos, Ypos)
    # # plt.xlabel('x (cm)')
    # # plt.ylabel('y (cm)')
    # # plt.show()

    # figPCA = plt.figure()
    # for ch in channel:
    #     angle = findXYwire_pca("20240226_STT_runs350_442.root" ,ch)
    #     wireAngle = np.append(wireAngle, angle)
    #     # wireYpos = np.append(wireYpos, Ypos)
    # plt.title('')
    # plt.show()
        
    output_dict = {}
    for l in range(0,3):
        layer_dict = {}
        for ch in channel[l]:
            hTracker, values = findXYZ(args.rootFile, args.coordFile, ch, plane[l])
            layer_dict[ch] = dict(zip(coord, values))
            print(ch, values)
        output_dict[layer[l]] = layer_dict
    output_dict["h0_hits"] = hTracker
    
    with open("measures.json", "w") as outfile: 
        json.dump(output_dict, outfile)