from matplotlib import pyplot as plt
import ROOT
import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from numpy.linalg import norm
import json
import re
# from numba import njit
import utils as utils

class Waveforms:
    def __init__(self, _channel):#, _chargeThr):
        self.Treepath = '/mnt/e/data/drift_chamber/' ##MODIFY ACCORDING TO LOCAL MACHINE PATH
        self.WFpath = '/mnt/e/data/drift_chamber/runs_350_onwards/' ##MODIFY ACCORDING TO LOCAL MACHINE PATH
        self.channel = "ch0"+re.split('c|_', _channel)[1]
        self.dig = "51" if len(re.split('c|_', _channel))>2 else "52"
        # self.chargeThr = _chargeThr
    def import2RDF(self, run_i, run_f, method, entry=0):
        """import waveforms to RootDataFrame and plot selected ones
        Args:
            run_i, run_f: initial-final run number
            method: "plot_entry", plot specifc entry/entries
                    "id", returns event id
            entry: int or array representing event numbers 
        """
        chain = ROOT.TChain("EventsFromDigitizer")
        pmchain = ROOT.TChain("EventsFromDigitizer")
        for r in range(run_i, run_f+1):
            chain.Add(self.WFpath+"run_"+str(r)+"_0x17"+self.dig+"0000v1751.root")
            pmchain.Add(self.WFpath+"run_"+str(r)+"_0x17520000v1751.root")
        df = ROOT.RDataFrame(chain)
        pmdf = ROOT.RDataFrame(pmchain)
        #self.dfBranches = df52.AsNumpy({"id", "ch00", "ch03"}) #this may not work (out of memory)
        # should filter directly over one entry and plot it, or work inside RDF
        match method:
            case "plot_WFentry":
                for e in entry:
                    chdf = df.Filter("id == " + str(e)).AsNumpy({self.channel})  
                    wf = [np.array(v) for v in chdf[self.channel]]
                    plt.plot(np.arange(0,len(wf[0])), np.asarray(wf[0]), linewidth=0.7)
                plt.xlabel("Iteration")
                plt.ylabel("signal")
                plt.show()
            case "plot_PMTentry":
                for e in entry:
                    chdf = pmdf.Filter("id == " + str(e)).AsNumpy({"ch07"})  
                    pm = [np.array(v) for v in chdf["ch07"]]
                    plt.plot(np.arange(0,len(pm[0])), np.asarray(pm[0]), linewidth=0.7)
                plt.xlabel("Iteration")
                plt.ylabel("signal")
                plt.show()
            case "id":
                dfBranch = df.AsNumpy({"id"})
                self.idList = [np.array(v) for v in dfBranch["id"]]
            
class TrackPosition:
    def __init__(self, _fname, _channel, _chargeThr, _tanThr):
        self.path = '/mnt/e/data/drift_chamber/'  ##MODIFY ACCORDING TO LOCAL MACHINE PATH
        self.fname = _fname
        self.channel = _channel
        self.chargeThr = _chargeThr 
        self.tanThr  = _tanThr
        self.WirePosition = np.empty(0)
        self.sigmaAngle = np.empty(0)
        self.WireAngle = np.empty(0)   
        self.WireCentroid = np.empty(0)      
    def import2RDF(self, filter_method):
        """import data from root TTree to RootDataFrame, and save as python list of arrays
        Args: filter method for data (charge or angle)
        """
        df = ROOT.RDataFrame("tree", self.path + self.fname)
        match filter_method:
            case "charge":
                dfBranches = df.Filter(self.channel + ">" + str(self.chargeThr)+
                                       "&&nTracksXZ==1&&nTracksYZ==1&&nClustersY==5&&nClustersX==5"
                                       ).AsNumpy({"x", "y", "z", "sx", "sy"})
                leafnames = ["x", "y", "z", "sx", "sy"]
            case "angle":
                dfBranches = df.Filter(self.channel + ">" + str(self.chargeThr) + "&&(sx>" + str(self.tanThr) + "||sx<" + str(-self.tanThr) + 
                                       ")&&(sy>" + str(self.tanThr) + "||sy<" + str(-self.tanThr)+")&&nTracksXZ==1&&nTracksYZ==1&&nClustersY==5&&nClustersX==5"
                                       ).AsNumpy({"x", "y", "z", "sx", "sy"})
                leafnames = ["x", "y", "z", "sx", "sy"]
        self.trackList = []
        for leaf in leafnames:
            self.trackList.append([np.array(v) for v in dfBranches[leaf]])               
    def readZmeasures(self, filename):
        """read measures between detector components from json file.
        Args: json file
        """
        f = open(filename)
        measures = json.load(f)[0]
        trk_3Wplane = measures["TRK_bottomDCScrews"] + measures["bottomDC_3WPLANE"]
        trk_2Wplane = trk_3Wplane + measures["3WPLANE_2WPLANE"]
        trk_1Wplane = trk_2Wplane + measures["2WPLANE_1WPLANE"]
        self.hplanes = [trk_1Wplane, trk_2Wplane, trk_3Wplane]
    def projectToWplane(self, height = 0):
        """Project hits at a certain heght wrt the one encoded in Branch 'z'.
        Args: height
        """
        self.projectedX = np.array(self.trackList[0]) + height*np.array(self.trackList[3])
        self.projectedY = np.array(self.trackList[1]) + height*np.array(self.trackList[4])
    def rotate(self, angle):
        """rotate all the hits wrt the origin of tracker SdR
        Args: angle
        """
        x, y = self.projectedX, self.projectedY
        cord = np.transpose(np.dstack((x,y)).reshape(-1,2))
        angle = angle * np.pi/180
        R = [[np.cos(angle), np.sin(angle)],[-np.sin(angle),np.cos(angle)]]
        new_cord = np.transpose(np.matmul(R,cord))
        self.rotated_x = new_cord[:,0]
        self.rotated_y = new_cord[:,1]
    def Gauss(self, x, A, x0, sigma):
        y = A*np.exp(-(x-x0)**2/(2*sigma**2))
        return y
    def fit_profile(self, plot=False):
        """fit hits trasversal profile (projected on Y axis) with gaussian"""
        fig = plt.figure()
        h2d = plt.hist2d(self.rotated_x, self.rotated_y, 240) #240 bins => 0.25cm/bin
        self.px_at_Theta = np.sum(h2d[0], axis=1)
        self.py_at_Theta = np.sum(h2d[0], axis=0)    
        self.posx_at_Theta = np.linspace(h2d[1][0] + (h2d[1][1]-h2d[1][0])/2, h2d[1][-2] + (h2d[1][-1]-h2d[1][-2])/2, len(self.px_at_Theta))
        self.posy_at_Theta = np.linspace(h2d[2][0] + (h2d[2][1]-h2d[2][0])/2, h2d[2][-2] + (h2d[2][-1]-h2d[2][-2])/2, len(self.py_at_Theta))
        self.pyPar_at_Theta, pcov = curve_fit(self.Gauss, self.posy_at_Theta, self.py_at_Theta, p0=[500, self.hplanes[1], 0.6])
        self.perr_at_Theta = np.sqrt(np.diag(pcov))
        # self.FWHM = curve_fit(np.arange(self.pyPar_at_Theta[1]-2, self.pyPar_at_Theta[1]+2, 0.01))==curve_fit(self.pyPar_at_Theta[1])/2
        # print(self.FWHM, self.pyPar_at_Theta[2])#TESTTT
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.close(fig)
        if plot:
            fig, ax = plt.subplots(1,2)
            ax[0].plot(self.posx_at_Theta, self.px_at_Theta, '.')
            ax[1].plot(self.posy_at_Theta, self.py_at_Theta, '.')
            ax[1].plot(self.posy_at_Theta, self.Gauss(self.posy_at_Theta, self.pyPar_at_Theta[0], self.pyPar_at_Theta[1], self.pyPar_at_Theta[2]), '-', label='sigma=%.3F' % self.pyPar_at_Theta[2])
            ax[0].set_xlabel('x profile')
            ax[1].set_xlabel('y profile')
            plt.legend()
            plt.show()
    def pca(self, plot=False):
        """perform pca"""
        hit_x = self.projectedX - np.mean(self.projectedX)
        hit_y = self.projectedY - np.mean(self.projectedY)
        hits_pos = np.array([hit_x, hit_y])
        _pca = PCA(n_components = 2).fit(np.transpose(hits_pos))
        self.comp1 = np.array([_pca.components_[0][0], _pca.components_[0][1]])
        self.comp2 = np.array([_pca.components_[1][0], _pca.components_[1][1]])
        weight1, weight2 = _pca.explained_variance_[0], _pca.explained_variance_[1]
        self.pca_angle = np.arccos(np.dot(self.comp1, np.array([1,0]))/(norm(self.comp1)))*180/np.pi
        if (self.pca_angle>90):
            self.pca_angle-=180
        if (((_pca.components_[0][0]>0)&(_pca.components_[0][1]<0))|((_pca.components_[0][0]<0)&(_pca.components_[0][1]<0))):
            self.pca_angle = -self.pca_angle
        self.centroid = np.array([np.mean(self.projectedX), np.mean(self.projectedY)])
        if plot:
            plt.plot(hits_pos[0,:], hits_pos[1,:], '.', markersize=2)
            for i, (comp, var) in enumerate(zip(_pca.components_, _pca.explained_variance_)):
                plt.plot([0, comp[0]], [0, comp[1]], label=f"Component {i}", color=f"C{i + 2}")
            plt.gca().set( title="2-dimensional dataset with principal components", xlabel="first feature", ylabel="second feature")
            plt.legend()
            plt.show()
    def fit_poly2d(self, x, lim, plot=False):
        """fit sigma distribution wrt X values with parabola
        Args: 
            x: x values (height, angle)
            lim: (int) fit limit
        """
        dydx = np.gradient(self.sigmaAngle)/self.sigmaAngle
        fit_dydx3 = np.polyfit(x, dydx, 3)
        dydx3_curve = np.poly1d(fit_dydx3)
        x0_idx = (np.abs(dydx3_curve(x) - 0)).argmin()
        x0 = x[x0_idx]    
        xFitRange = x[(x > x0-lim)&(x < x0+lim)]
        sigmaFitRange = self.sigmaAngle[(x > x0-lim)&(x < x0+lim)]
        sigmaFit = np.polyfit(xFitRange, sigmaFitRange, 2)
        sigmaCurve = np.poly1d(sigmaFit)
        self.minH = xFitRange[np.array(sigmaCurve(xFitRange)) == np.min(np.array(sigmaCurve(xFitRange)))][0]
        self.minSigma = np.min(np.array(sigmaCurve(xFitRange)))
        self.minAngle = self.WireAngle[np.array(sigmaCurve(x)) == np.min(np.array(sigmaCurve(x)))][0]
        if plot:
            plt.plot(x, self.sigmaAngle, '.')
            plt.plot(xFitRange, sigmaCurve(xFitRange), '--')
            plt.axvline(self.minH, color='g', linestyle='--', label=self.channel+" h="+str(self.minH))
            # plt.xlabel('height (cm)')
            plt.ylabel('sigma')
            plt.legend()
            plt.show()    
    def plotTracks(self, plottype):
        match plottype:
            case "original":
                plt.plot(self.trackList[0], self.trackList[1], '.', markersize=2, label='original')
            case "projected":
                plt.plot(self.projectedX, self.projectedY, '.', markersize=2, label='projected')
            case "rotated":
                plt.plot(self.rotated_x, self.rotated_y, '.', markersize=2, label='rotated')
            case "wireFit":
                xline = np.linspace(0,40,40)*np.cos(self.minAngle* np.pi/180)
                yline = float(self.pyPar_at_Theta[1]) + np.linspace(0,40,40)*np.sin(self.minAngle* np.pi/180)
                plt.plot(self.projectedX, self.projectedY, '.', markersize=2)
                plt.plot(xline, yline, '--', label=r': $\theta$='+ '{:.3f}'.format(self.minAngle)+r'° , $\sigma$='+ '{:.2f}'.format(self.pyPar_at_Theta[2]))
                plt.plot(self.centroid[0], self.centroid[1], 'o')
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.title(self.channel)
        
class TimeDistance:
    def __init__(self, _fname, _channel, _chargeThr):
        self.path = '/mnt/e/data/drift_chamber/'  ##MODIFY ACCORDING TO LOCAL MACHINE PATH
        self.fname = _fname
        self.channel = _channel
        self.chargeThr = _chargeThr  
    def import2RDF(self):
        df = ROOT.RDataFrame("tree", self.path + self.fname)
        dfTrackBranches = df.Filter(self.channel + "_c>" + str(self.chargeThr)+"&&"+str(self.channel)+"_pp>0.015&&pm7_c>0.2e-10&&nTracksXZ==1&&nTracksYZ==1&&nClustersY==5&&nClustersX==5"
                                                                                ).AsNumpy({"entry", "x", "y", "z", "sx", "sy"})
        dfTimeBranches = df.Filter(self.channel + "_c>" + str(self.chargeThr)+"&&"+str(self.channel)+"_pp>0.015&&pm7_c>0.2e-10&&nTracksXZ==1&&nTracksYZ==1&&nClustersY==5&&nClustersX==5"
                                                                                ).AsNumpy({str(self.channel)+"_t", "pm7_t"})
        trackLeaf = ["x", "y", "z", "sx", "sy"]
        TimeLeaf = [str(self.channel)+"_t", "pm7_t"]
        self.trackList = []
        self.TimeList = []
        self.entryList = [np.array(v) for v in dfTrackBranches["entry"]]
        for leaf in trackLeaf:
            self.trackList.append([np.array(v) for v in dfTrackBranches[leaf]])
        for leaf in TimeLeaf:
            self.TimeList.append([np.array(v) for v in dfTimeBranches[leaf]])
    def projectToWplane(self, height = 0):
        self.projectedX = np.array(self.trackList[0]) + height*np.array(self.trackList[3])
        self.projectedY = np.array(self.trackList[1]) + height*np.array(self.trackList[4])
    def compute_distance3D(self, Wpoint, Wangle):  
        """compute distance using track and wire equations
        Args:
            Wpoint: point of the wire, coordinates [x,y,z]
            Wangle: wire angle (deg)
        """
        tanX, tanY = np.array(self.trackList[3]), np.array(self.trackList[4])
        Wvec = np.transpose(np.array([np.cos(Wangle*np.pi/180), np.sin(Wangle*np.pi/180), 0]).reshape(3,1) *np.ones((1, tanX.size)) )
        Tvec = np.transpose(np.array([tanX, tanY, np.ones(tanX.size)]))
        Wpoint = Wpoint.reshape(1,3)
        Tpoint = np.array([self.projectedX, self.projectedY, np.zeros((self.projectedY).size)]).T
        # NOTE: I assume z=0 for points of both wire and tracks
        self.distance = np.abs(np.diag(np.matmul(np.cross(Wvec, Tvec),np.transpose(Tpoint-Wpoint))))

class Distance:
    def __init__(self, _fname, _channel, _chargeThr):
        self.path = '/mnt/e/data/drift_chamber/'  ##MODIFY ACCORDING TO LOCAL MACHINE PATH
        self.fname = _fname
        self.channel = _channel#"dc"+str(_channel)
        self.chargeThr = _chargeThr  

    def import2RDF(self):
        df = ROOT.RDataFrame("tree", self.path + self.fname)
        cluster_filter = self.channel + "_c>" + str(self.chargeThr)+"&&nTracksXZ==1&&nTracksYZ==1&&nClustersY==5&&nClustersX==5"
        sigma_filter = "sigma>0.7"
        
        trk = df.Filter(cluster_filter).Define(
                "m_fit", "Numba::fit_m(clY_z, clX_pos.clY_pos)"
                ).Define(
                "q_fit", "Numba::fit_q(clY_z, clX_pos.clY_pos, m_fit)"
                ).Define(
                "sigma", "Numba::sigma_fit(clY_z, clX_pos.clY_pos, m_fit, q_fit)"
                )
        new_trk = trk.Define(
            "new_fit", "Numba::find_best_fit(clY_z, clX_pos.clY_pos, m_fit, q_fit, sigma)"
            )
        
        # SAVE NEW RDF TO ROOT FILE, to be read in TimeDistance class
        
        plot_trk = True
        if plot_trk:
            trk1 = trk.Filter(sigma_filter).AsNumpy({"clY_z", "clX_pos.clY_pos", "m_fit", "q_fit"}) 
            m = [np.array(v) for v in trk1["m_fit"]]
            q = [np.array(v) for v in trk1["q_fit"]]
            pos = [np.array(v) for v in trk1["clX_pos.clY_pos"]]
            cly = [np.array(v) for v in trk1["clY_z"]]
            pos = np.array(pos)
            cly = np.array(cly)
            m, q = np.array(m), np.array(q)
            
            new_trk1 = new_trk.Filter(sigma_filter).AsNumpy({"new_fit"}) 
            new_par = [np.array(v) for v in new_trk1["new_fit"]]
            new_par = np.concatenate(new_par).reshape(-1,4)
        
            plt.plot(pos, cly, '.', markersize=2)
            idx = np.arange(0,5)
            for i in range(0,6):
                dots = plt.plot(pos[i].T, cly[i].T, 'o')
                color = dots[0].get_color()
                #plt.plot([m[i]*cly[i,0]+q[i], m[i]*cly[i,-1]+q[i]], [cly[i,0], cly[i,-1]], linestyle='--', color=color)
                new_cly = np.take(cly[i], np.where(idx!=new_par[i,3])[0])
                plt.plot([new_par[i,0]*new_cly[0]+new_par[i,1], new_par[i,0]*new_cly[-1]+new_par[i,1]], [new_cly[0], new_cly[-1]], linestyle='--', color=color)
            plt.ylabel('h (cm)')
            plt.xlabel('pos Y (cm)')
            plt.show()
        
        plot_distr = False
        if plot_distr:
            trk2 = trk.AsNumpy({"m_fit", "sigma"})        
            new_trk2 = new_trk.AsNumpy({"new_fit"})  
            m = [np.array(v) for v in trk2["m_fit"]]
            m = np.array(m)
            sigma = [np.array(v) for v in trk2["sigma"]]
            sigma = np.array(sigma)
            new_par = [np.array(v) for v in new_trk2["new_fit"]]
            new_par = np.concatenate(new_par).reshape(-1,4)
            plt.figure("sigma")
            bin_sigma1 = 200
            bin_sigma2 = int(bin_sigma1 * (np.max(new_par[:,2])-np.min(new_par[:,2]))/(np.max(sigma)-np.min(sigma)))
            #plt.hist(sigma, bin_sigma1, histtype='step', label='old')
            plt.hist(new_par[:,2], bin_sigma2, histtype='step', label='new')
            plt.yscale('log')
            plt.xlabel('sigma (cm)')
            # plt.legend()
            plt.figure("Ytangent")
            bin_tan1 = 200
            bin_tan2 = int(bin_tan1 * (np.max(new_par[:,0])-np.min(new_par[:,0]))/(np.max(m)-np.min(m)))
            plt.hist(m, bin_tan1, histtype='step', label='old')
            plt.hist(new_par[:,0], bin_tan2, histtype='step', label='new')
            plt.yscale('log')
            plt.xlabel('tan YZ')
            plt.legend()
            plt.show()