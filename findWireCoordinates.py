from matplotlib import pyplot as plt
import numpy as np
from waveforms import Tracks

def compute_distance(dot_x, dot_y, line_x, line_y):
    p1 = np.array([line_x[0], line_y[0]])
    p2 = np.array([line_x[-1], line_y[-1]])
    p3 = np.array([dot_x, dot_y])
    distance = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
    return distance

def find_wire(channel, plot=False):
    wirePos = np.empty(0)
    sigma = np.empty(0)
    angle = np.arange(-11,5,0.2)
    tracks = Tracks("20240226_STT_runs350_442.root", channel, 0.02e-9) #20240220_STT.root
    tracks.import2RDF()

    for t in angle:
        tracks.rotate(t)
        tracks.build_profile()
        wirePos = np.append(wirePos, tracks.pyPar_at_Theta[1])
        sigma = np.append(sigma, abs(tracks.pyPar_at_Theta[2]))

    sigmaFit = np.polyfit(angle, sigma, 4)
    sigmaCurve = np.poly1d(sigmaFit)
    minimumAngle = angle[np.array(sigmaCurve(angle)) == np.min(np.array(sigmaCurve(angle)))][0]
    tracks.rotate(minimumAngle)
    tracks.build_profile()
    wirePosition = tracks.pyPar_at_Theta[1]

    xline = np.linspace(0,40,40)*np.cos(minimumAngle* np.pi/180)
    yline = float(wirePosition) + np.linspace(0,40,40)*np.sin(minimumAngle* np.pi/180)
    plt.plot(tracks.trackList[0],tracks.trackList[1], '.', markersize=2)
    plt.plot(xline, yline, '--', label=channel[:3]+r': $\theta$='+ '{:.3f}'.format(minimumAngle)+r'° , $\sigma$='+ '{:.2f}'.format(tracks.pyPar_at_Theta[2]))
    plt.legend()
    
    if plot:
        tracks.rotate(minimumAngle)
        tracks.build_profile(True)
        
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
    
    return 0#minimumAngle, wirePosition

def find_wire_PCA(channel, plotPCA=False, plotProfile=False):
    tracksPCA = Tracks("20240226_STT_runs350_442.root", channel, 0.02e-9) #20240220_STT.root
    tracksPCA.import2RDF()

    tracksPCA.pca(plotPCA)

    minimumAngle = tracksPCA.pca_angle
    tracksPCA.rotate(minimumAngle)
    tracksPCA.build_profile(plotProfile)
    wirePosition = tracksPCA.pyPar_at_Theta[1]

    xline = np.linspace(0,40,40)*np.cos(minimumAngle* np.pi/180)
    yline = float(wirePosition) + np.linspace(0,40,40)*np.sin(minimumAngle* np.pi/180)
    plt.plot(tracksPCA.trackList[0],tracksPCA.trackList[1], '.', markersize=2)
    plt.plot(xline, yline, '--', label=channel[:3]+r': $\theta$='+ '{:.3f}'.format(minimumAngle)+r'° , $\sigma$='+ '{:.2f}'.format(tracksPCA.pyPar_at_Theta[2]))
    plt.legend()  
    
    return 0#minimumAngle, wirePosition

if __name__ == "__main__":
    
    wireYpos = np.empty(0)
    wireAngle = np.empty(0)
    channel = ["dc0_c", "dc1_c", "dc2_c", "dc3_c", "dc4_c", "dc5_c"] 
    
    fig = plt.figure()
    for ch in channel:
        find_wire(ch)
        # wireAngle = np.append(wireAngle, angle)
        # wireYpos = np.append(wireYpos, Ypos)
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()
    
    figPCA = plt.figure()
    for ch in channel:
        find_wire_PCA(ch)
        # wireAngle = np.append(wireAngle, angle)
        # wireYpos = np.append(wireYpos, Ypos)
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()