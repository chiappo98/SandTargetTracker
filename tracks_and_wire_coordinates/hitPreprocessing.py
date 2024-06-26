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
            case "new_tracks":
                dfBranches = df.Filter(self.channel + ">" + str(self.chargeThr)).AsNumpy({"newX", "newY", "z1", "tx", "ty"}) #, "Xindex", "Yindex"
                leafnames = ["newX", "newY", "z1", "tx", "ty"] #, "Xindex", "Yindex"
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
        self.newX = np.array(self.trackList[0]) + height*np.array(self.trackList[3])
        self.newY = np.array(self.trackList[1]) + height*np.array(self.trackList[4])
    def rotate(self, angle):
        """rotate all the hits wrt the origin of tracker SdR
        Args: angle
        """
        x, y = self.newX, self.newY
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
        hit_x = self.newX - np.mean(self.newX)
        hit_y = self.newY - np.mean(self.newY)
        hits_pos = np.array([hit_x, hit_y])
        _pca = PCA(n_components = 2).fit(np.transpose(hits_pos))
        #_var = _pca.get_covariance()
        #print(_cov)
        self.cov1 = np.array([_pca.get_covariance()[0][0], _pca.get_covariance()[0][1]])
        self.comp1 = np.array([_pca.components_[0][0], _pca.components_[0][1]])
        self.comp2 = np.array([_pca.components_[1][0], _pca.components_[1][1]])
        #weight1, weight2 = _pca.explained_variance_[0], _pca.explained_variance_[1]
        self.pca_angle = np.arccos(np.dot(self.comp1, np.array([1,0]))/(norm(self.comp1)))*180/np.pi
        if (self.pca_angle>90):
            self.pca_angle-=180
        if (((_pca.components_[0][0]>0)&(_pca.components_[0][1]<0))|((_pca.components_[0][0]<0)&(_pca.components_[0][1]<0))):
            self.pca_angle = -self.pca_angle
        self.centroid = np.array([np.mean(self.newX), np.mean(self.newY)])
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
        # compute first derivative of the curve
        dydx = np.gradient(self.sigmaAngle)/self.sigmaAngle
        # fit the derivative excluding bad points
        fit_dydx = np.polyfit(x[(dydx>-0.1)&(dydx<0.1)], dydx[(dydx>-0.1)&(dydx<0.1)], 3)
        dydx_curve = np.poly1d(fit_dydx)        
        # compute the range of the fit for data points
        x0_idx = (np.abs(dydx_curve(x) - 0)).argmin()
        x0 = x[x0_idx]    
        # exclude bad points from fit
        xFitRange = x[((x > x0-lim)&(x < x0+lim))&(self.sigmaAngle>0.4)]
        sigmaFitRange = self.sigmaAngle[((x > x0-lim)&(x < x0+lim))&(self.sigmaAngle>0.4)]
        sigmaFit = np.polyfit(xFitRange, sigmaFitRange, 2, cov=True)
        par_sigma = sigmaFit[0]
        var_sigma = np.diag(sigmaFit[1])
        sigmaCurve = np.poly1d(par_sigma)
        minh = -par_sigma[1]/(2*par_sigma[0])
        index = (np.abs(x - minh)).argmin()
        self.minH =  x[index]
        self.minH_err = minh*0.5*np.sqrt( ( var_sigma[0]/(par_sigma[0]**2) + var_sigma[1]/(par_sigma[1]**2)) )
        self.minSigma = sigmaCurve(self.minH)
        self.minAngle = self.WireAngle[x == self.minH][0]
        if plot:
            fig, ax1 = plt.subplots()
            ax1.plot(x, self.sigmaAngle, '.')
            ax1.plot(xFitRange, sigmaCurve(xFitRange), '--')
            ax2 = ax1.twinx() 
            ax2.plot(x, dydx, '--')
            ax1.axvline(self.minH, color='g', linestyle='--', label=self.channel+" h="+str(self.minH))
            ax1.set_ylabel('sigma')
            ax1.legend()
            plt.show()    
    def plotTracks(self, plottype):
        match plottype:
            case "original":
                plt.plot(self.trackList[0], self.trackList[1], '.', markersize=2, label='original')
            case "projected":
                plt.plot(self.newX, self.newY, '.', markersize=2, label='projected')
            case "rotated":
                plt.plot(self.rotated_x, self.rotated_y, '.', markersize=2, label='rotated')
            case "wireFit":
                xline = np.linspace(0,40,40)*np.cos(self.minAngle* np.pi/180)
                yline = float(self.pyPar_at_Theta[1]) + np.linspace(0,40,40)*np.sin(self.minAngle* np.pi/180)
                plt.plot(self.newX, self.newY, '.', markersize=2)
                plt.plot(xline, yline, '--', label=r': $\theta$='+ '{:.3f}'.format(self.minAngle)+r'° , $\sigma$='+ '{:.2f}'.format(self.pyPar_at_Theta[2]))
                plt.plot(self.centroid[0], self.centroid[1], 'o')
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.title(self.channel)
        
class TimeDistance:
    def __init__(self, _fname, _channel, _tan_thr):
        self.path = '/mnt/e/data/drift_chamber/'  ##MODIFY ACCORDING TO LOCAL MACHINE PATH
        self.fname = _fname
        self.channel = _channel 
        self.tan_thr1 = _tan_thr[0]
        self.tan_thr2 = _tan_thr[1]
    def import2RDF(self):
        df = ROOT.RDataFrame("tree", self.path + self.fname)
        channels = np.array( [["dc0", "dc3", "dc0_1", "dc3_1"], ["dc1", "dc4", "dc1_1", "dc4_1"], ["dc2", "dc5", "dc2_1", "dc5_1"]] )
        idx = np.where(channels==(self.channel))[1][0]
        excluded_ch = channels[(channels!=channels[0,idx])&(channels!=channels[1,idx])&(channels!=channels[2,idx])]
        excluded_str = "_cf<20e-12&&".join((excluded_ch).tolist())
        df_filt = df.Filter(self.channel + "_cf>20e-12&&pm7_c>10e-12&&ty>"+str(self.tan_thr1)+"&&ty<"+str(self.tan_thr2)+"&&"+excluded_str+'_cf<20e-12')     
        dfTrackBranches = df_filt.AsNumpy({"entry", "newX", "newY", "z1", "tx", "ty"}) #{"entry", "x", "y", "z", "sx", "sy"})
        dfTimeBranches = df_filt.AsNumpy({str(self.channel)+"_tf", "pm7_t"}) #tf
        trackLeaf =  ["newX", "newY", "z1", "tx", "ty"] #"x", "y", "z", "sx", "sy"
        TimeLeaf = [str(self.channel)+"_tf", "pm7_t"] #tf
        self.trackList = []
        self.TimeList = []
        self.entryList = [np.array(v) for v in dfTrackBranches["entry"]]
        for leaf in trackLeaf:
            self.trackList.append([np.array(v) for v in dfTrackBranches[leaf]])
        for leaf in TimeLeaf:
            self.TimeList.append([np.array(v) for v in dfTimeBranches[leaf]])
    def projectToWplane(self, height = 0):
        self.projX = np.array(self.trackList[0]) + height*np.array(self.trackList[3])
        self.projY = np.array(self.trackList[1]) + height*np.array(self.trackList[4])
    def compute_distance3D(self, Wpoint, Wangle):  
        """compute distance using track and wire equations
        Args:
            Wpoint: point of the wire, coordinates [x,y,z]
            Wangle: wire angle (deg)
        """
        tanX, tanY = np.array(self.trackList[3]), np.array(self.trackList[4])
        Wvec = (np.array([np.cos(Wangle*np.pi/180), np.sin(Wangle*np.pi/180), 0]).reshape(3,1) *np.ones((1, tanX.size)) ).T
        Tvec = (np.array([tanX, tanY, np.ones(tanX.size)])).T
        Wpoint = Wpoint.reshape(1,3).T
        Tpoint = np.array([self.projX, self.projY, np.zeros((self.projX).size)])
        # NOTE: I assume z=0 for points of both wire and tracks
        # Filter: keep only tracks inside the cell
        top_position = Tpoint + 0.5*Tvec.T
        bottom_position = Tpoint - 0.5*Tvec.T
        def wire_y(x):
            return np.tan(Wangle*np.pi/180) * (x - Wpoint[0]) + Wpoint[1]
        cell_filter = ((top_position[1,:] < (wire_y(top_position[0,:])+1)) & (top_position[1,:] > (wire_y(top_position[0,:])-1))) & ((bottom_position[1,:] < (wire_y(bottom_position[0,:])+1)) & (bottom_position[1,:] > (wire_y(bottom_position[0,:])-1)))
        # Compute 3D distance
        direction3d = np.cross(Wvec, Tvec).T
        direction3d/=np.linalg.norm(direction3d, axis=0)
        self.distance = np.abs(np.diag(np.matmul((Tpoint-Wpoint).T, direction3d)))[cell_filter]
        # Compute 2D distance
        a = 1
        b = -1/np.tan(Wangle*np.pi/180)
        c = (Wpoint[1]/np.tan(Wangle*np.pi/180)) - Wpoint[0]
        self.distance2D = ((a*Tpoint[0] + b*Tpoint[1] + c) / np.sqrt( a**2 + b**2))[cell_filter]
        self.TimeList[0] = np.array(self.TimeList[0])[cell_filter]
        self.TimeList[1] = np.array(self.TimeList[1])[cell_filter]
        
        self.ty = np.array(self.trackList[4])[cell_filter]
        
    def time_calibration(self, time_delay):
        def t0_func(t, t0, t1, a, b, c, d):
            y = a + b*(np.exp(-d*(t-t1))/(1+np.exp(-(t-t0)/c)))
            return y
        t0 = np.array(self.TimeList[0])
        t1 = np.array(self.TimeList[1]) 
        dt = t0-t1-time_delay
        plt.figure()
        n, bins, patches = plt.hist(dt[(dt>-0.1e-7)&(dt<1.3e-7)], 100)
        bin_to_fit = (bins[:-1] + (bins[1]-bins[0])/2)
        ppar, pcov = curve_fit(t0_func, bin_to_fit[bin_to_fit<0.8e-7], n[bin_to_fit<0.8e-7], bounds=([-2e-8, -1e-7, 0, 0, 0, 1e6],[0.5e-7, 1e-7, 3, 2e3, 5e-9, 12e6]), maxfev=5000)
        plt.plot(bin_to_fit[bin_to_fit<0.8e-7], t0_func(bin_to_fit[bin_to_fit<0.8e-7], *ppar))
        plt.plot(ppar[0], t0_func(ppar[0], *ppar), '*')
        plt.title(self.channel)
        # plt.show()
        plt.close()

        self.DeltaT = dt-ppar[0]
        

class Tracks:
    def __init__(self, _fname, _channel, _chargeThr):
        self.path = '/mnt/e/data/drift_chamber/'  ##MODIFY ACCORDING TO LOCAL MACHINE PATH
        self.fname = _fname
        self.channel = _channel
        self.chargeThr = _chargeThr  
    def import2RDF(self, method):
        df = ROOT.RDataFrame("tree", self.path + self.fname)
        cluster_filter = self.channel + "_c>" + str(self.chargeThr)+"&&nTracksXZ==1&&nTracksYZ==1&&nClustersY==5&&nClustersX==5"
        all_wires_filter = "nTracksXZ==1&&nTracksYZ==1&&nClustersY==5&&nClustersX==5"
        sigma_filter = "sigmaX>0.7&&sigmaY>0.7" 
        index_filter = "Xindex!=3&&Yindex!=3&&Xindex!=-4&&Yindex!=-4"  #HAVE TO ACCOUNT ALSO FOR EXCLUDED DOTS (NOT EXCLUDING THEM!)
        tan_filter = "tx>-0.4&&tx<0.4&&ty>-0.4&&ty<0.4"
        
        match method:
            case "all_wires":
                method_filter = all_wires_filter
                plot_old_trk = False
                plot_double = False
                plot_largeM = False
                plot_distr = True
                plot_xy = False
                saveRDF = True
            case "single_wire":
                method_filter = cluster_filter
                plot_old_trk = False
                #plot_new_trk = False
                plot_double = False
                plot_largeM = False
                plot_distr = True
                plot_xy = True
                saveRDF = False
        
        trk = df.Filter(method_filter).Define(
                "fitX", "Numba::fit_track(clX_z, clX_pos)"
                ).Define(
                "fitY", "Numba::fit_track(clY_z, clY_pos)"
                ).Define(
                "sigmaX", "Numba::select_col(fitX, 2)"
                ).Define(
                "sigmaY", "Numba::select_col(fitY, 2)"
                )
        # new_trk = trk.Define(
        #         "new_fitX", "Numba::find_best_fit(clX_z, clX_pos.clY_pos, fitX)"
        #         ).Define(
        #         "new_fitY", "Numba::find_best_fit(clY_z, clY_pos, fitY)"
        #         )
        double_trk = trk.Define(
                "double_fitX", "Numba::find_best_fit2(clX_z, clX_pos, fitX)"
                ).Define(
                "double_fitY", "Numba::find_best_fit2(clY_z, clY_pos, fitY)"
                )
        proj_trk = double_trk.Define(
            "tx", "Numba::select_col(double_fitX, 1)"
            ).Define(
            "ty", "Numba::select_col(double_fitY, 1)"
            ).Define(
            "double_qX", "Numba::select_col(double_fitX, 3)"
            ).Define(
            "double_qY", "Numba::select_col(double_fitY, 3)"
            ).Define(
            "newX", "Numba::project_track(z, tx, double_qX)"
            ).Define(
            "newY", "Numba::project_track(z, ty, double_qY)"
            ).Define(
            "Xindex", "Numba::select_col(double_fitX, 5)"
            ).Define(
            "Yindex", "Numba::select_col(double_fitY, 5)"
            ).Define(
                "z1",  "Numba::redefine_z(z)")
        if saveRDF:
            proj_trk.Snapshot("tree", self.path+"tracks_20240501_STT_run2024_2365.root",
                              {"entry", "dc0_cf", "dc3_cf", "dc0_1_cf", "dc3_1_cf", 
                               "dc1_cf", "dc4_cf", "dc1_1_cf", "dc4_1_cf", 
                               "dc2_cf", "dc5_cf", "dc2_1_cf", "dc5_1_cf", "pm7_c",
                               "dc0_tf", "dc3_tf", "dc0_1_tf", "dc3_1_tf", 
                               "dc1_tf", "dc4_tf", "dc1_1_tf", "dc4_1_tf", 
                               "dc2_tf", "dc5_tf", "dc2_1_tf", "dc5_1_tf", "pm7_t",
                               "x", "y", "z1", "tx", "ty", "newX", "newY"});    
        if plot_xy:
            proj_trk1 = proj_trk.Filter(index_filter).AsNumpy({"newX", "newY", "tx", "ty", "x", "y", "sx", "sy"})
            proj_x = [np.array(v) for v in proj_trk1["newX"]]
            proj_y = [np.array(v) for v in proj_trk1["newY"]]
            tan_x = [np.array(v) for v in proj_trk1["tx"]]
            tan_y = [np.array(v) for v in proj_trk1["ty"]]
            x = [np.array(v) for v in proj_trk1["x"]]
            y = [np.array(v) for v in proj_trk1["y"]]
            sx = [np.array(v) for v in proj_trk1["sx"]]
            sy = [np.array(v) for v in proj_trk1["sy"]]
            self.proj_x = np.array(proj_x)
            self.proj_y = np.array(proj_y)
            self.tan_x = np.array(tan_x)
            self.tan_y = np.array(tan_y)
            self.x = np.array(x)
            self.y = np.array(y)
            self.sx = np.array(sx)
            self.sy = np.array(sy) 
            
## Plot tracks and distributions

        run_i, run_f = 0,7

        if plot_old_trk:
            trk1 = trk.Filter(sigma_filter).AsNumpy({"clX_z", "clY_z", "clX_pos", "clY_pos", "fitX", "fitY"})    
                     
            posx = [np.array(v) for v in trk1["clX_pos"]]
            posy = [np.array(v) for v in trk1["clY_pos"]]
            clx = [np.array(v) for v in trk1["clX_z"]]
            cly = [np.array(v) for v in trk1["clY_z"]]
            parx = [np.array(v) for v in trk1["fitX"]]
            pary = [np.array(v) for v in trk1["fitY"]]
            posx = np.array(posx)
            posy = np.array(posy)
            clx = np.array(clx)
            cly = np.array(cly)
            parx = np.concatenate(parx).reshape(-1,3)
            pary = np.concatenate(pary).reshape(-1,3)
            
            plt.figure("x view")
            plt.plot(posx, clx, '.', markersize=2)
            for i in range(run_i, run_f):
                dots = plt.plot(posx[i].T, clx[i].T, 'o')
                color = dots[0].get_color()
                plt.plot([parx[i,0]*clx[i,0]+parx[i,1], parx[i,0]*clx[i,-1]+parx[i,1]], [clx[i,0], clx[i,-1]], linestyle='--', color=color)
            plt.ylabel('h (cm)')
            plt.xlabel('pos X (cm)')
            
            plt.figure("y view")
            plt.plot(posy, cly, '.', markersize=2)
            for i in range(run_i, run_f):
                dots = plt.plot(posy[i].T, cly[i].T, 'o')
                color = dots[0].get_color()
                plt.plot([pary[i,0]*cly[i,0]+pary[i,1], pary[i,0]*cly[i,-1]+pary[i,1]], [cly[i,0], cly[i,-1]], linestyle='--', color=color)
            plt.ylabel('h (cm)')
            plt.xlabel('pos Y (cm)')
            plt.show()
            
        # if plot_new_trk:
        #     new_trk1 = new_trk.Filter(sigma_filter).AsNumpy({"clY_z", "clX_pos.clY_pos", "new_fit"}) 
            
        #     pos = [np.array(v) for v in new_trk1["clX_pos.clY_pos"]]
        #     cly = [np.array(v) for v in new_trk1["clY_z"]]
        #     new_par = [np.array(v) for v in new_trk1["new_fit"]]
        #     pos = np.array(pos)
        #     cly = np.array(cly)
        #     new_par = np.concatenate(new_par).reshape(-1,3)
        
        #     plt.plot(pos, cly, '.', markersize=2)
        #     for i in range(run_i, run_f):
        #         dots = plt.plot(pos[i].T, cly[i].T, 'o')
        #         color = dots[0].get_color()
        #         plt.plot([new_par[i,0]*cly[i,0]+new_par[i,1], new_par[i,0]*cly[i,-1]+new_par[i,1]], [cly[i,0], cly[i,-1]], linestyle='--', color=color)
        #     plt.ylabel('h (cm)')
        #     plt.xlabel('pos Y (cm)')
        #     plt.show()
        
        if plot_double:
            double_trk1 = double_trk.Filter(sigma_filter).AsNumpy({"clX_z", "clY_z", "clX_pos", "clY_pos", "double_fitX", "double_fitY"}) 
            
            posx = [np.array(v) for v in double_trk1["clX_pos"]]
            posy = [np.array(v) for v in double_trk1["clY_pos"]]
            clx = [np.array(v) for v in double_trk1["clX_z"]]
            cly = [np.array(v) for v in double_trk1["clY_z"]]
            new_parx = [np.array(v) for v in double_trk1["double_fitX"]]
            new_pary = [np.array(v) for v in double_trk1["double_fitY"]]
            posx = np.array(posx)
            posy = np.array(posy)
            clx = np.array(clx)
            cly = np.array(cly)
            new_parx = np.concatenate(new_parx).reshape(-1,6)
            new_pary = np.concatenate(new_pary).reshape(-1,6)
        
            plt.figure("x view")
            plt.plot(posx, clx, '.', markersize=2)
            for i in range(run_i, run_f):
                dots = plt.plot(posx[i].T, clx[i].T, 'o')
                color = dots[0].get_color()
                if new_parx[i,0] == new_parx[i,1]:
                    plt.plot([new_parx[i,0]*clx[i,0]+new_parx[i,2], new_parx[i,0]*clx[i,-1]+new_parx[i,2]], [clx[i,0], clx[i,-1]], linestyle='--', color=color)
                else:
                    x1, x2 = clx[i, :int(new_parx[i,5])+1], clx[i, int(new_parx[i,5]):]
                    plt.plot([new_parx[i,0]*x1[0]+new_parx[i,2], new_parx[i,0]*x1[-1]+new_parx[i,2]], [x1[0], x1[-1]], linestyle='--', color=color)
                    plt.plot([new_parx[i,1]*x2[0]+new_parx[i,3], new_parx[i,1]*x2[-1]+new_parx[i,3]], [x2[0], x2[-1]], linestyle='--', color=color)           
            plt.ylabel('h (cm)')
            plt.xlabel('pos X (cm)')
            
            plt.figure("y view")
            plt.plot(posy, cly, '.', markersize=2)
            for i in range(run_i, run_f):
                dots = plt.plot(posy[i].T, cly[i].T, 'o')
                color = dots[0].get_color()
                if new_pary[i,0] == new_pary[i,1]:
                    plt.plot([new_pary[i,0]*cly[i,0]+new_pary[i,2], new_pary[i,0]*cly[i,-1]+new_pary[i,2]], [cly[i,0], cly[i,-1]], linestyle='--', color=color)
                else:
                    x1, x2 = cly[i, :int(new_pary[i,5])+1], cly[i, int(new_pary[i,5]):]
                    plt.plot([new_pary[i,0]*x1[0]+new_pary[i,2], new_pary[i,0]*x1[-1]+new_pary[i,2]], [x1[0], x1[-1]], linestyle='--', color=color)
                    plt.plot([new_pary[i,1]*x2[0]+new_pary[i,3], new_pary[i,1]*x2[-1]+new_pary[i,3]], [x2[0], x2[-1]], linestyle='--', color=color)           
            plt.ylabel('h (cm)')
            plt.xlabel('pos Y (cm)')
            plt.show()
            
        if plot_largeM:
            m_filter = "ty<-0.2|ty>0.2" #"tx<-0.4|tx>0.4|ty<-0.4|ty>0.4"
            ext_trk = double_trk.Define(
                "double_sigmaX", "Numba::select_col(double_fitX, 4)"
                ).Define(
                "double_sigmaY", "Numba::select_col(double_fitY, 4)"
                ).Define(
                "tx", "Numba::select_col(double_fitX, 1)"
                ).Define(
                "ty", "Numba::select_col(double_fitY, 1)"
                ).Filter(m_filter).AsNumpy({"clX_z", "clY_z", "clX_pos", "clY_pos", "double_fitX", "double_fitY"}) 
                    
            posx = [np.array(v) for v in ext_trk["clX_pos"]]
            posy = [np.array(v) for v in ext_trk["clY_pos"]]
            clx = [np.array(v) for v in ext_trk["clX_z"]]
            cly = [np.array(v) for v in ext_trk["clY_z"]]
            new_parx = [np.array(v) for v in ext_trk["double_fitX"]]
            new_pary = [np.array(v) for v in ext_trk["double_fitY"]]
            posx = np.array(posx)
            posy = np.array(posy)
            clx = np.array(clx)
            cly = np.array(cly)
            new_parx = np.concatenate(new_parx).reshape(-1,6)
            new_pary = np.concatenate(new_pary).reshape(-1,6)
        
            plt.figure("x view")
            # plt.plot(posx, clx, '.', markersize=2)
            for i in range(run_i, run_f):
                dots = plt.plot(posx[i].T, clx[i].T, 'o')
                color = dots[0].get_color()
                if new_parx[i,0] == new_parx[i,1]:
                    plt.plot([new_parx[i,0]*clx[i,0]+new_parx[i,2], new_parx[i,0]*clx[i,-1]+new_parx[i,2]], [clx[i,0], clx[i,-1]], linestyle='--', color=color)
                else:
                    x1, x2 = clx[i, :int(new_parx[i,5])+1], clx[i, int(new_parx[i,5]):]
                    plt.plot([new_parx[i,0]*x1[0]+new_parx[i,2], new_parx[i,0]*x1[-1]+new_parx[i,2]], [x1[0], x1[-1]], linestyle='--', color=color)
                    plt.plot([new_parx[i,1]*x2[0]+new_parx[i,3], new_parx[i,1]*x2[-1]+new_parx[i,3]], [x2[0], x2[-1]], linestyle='--', color=color)           
            plt.ylabel('h (cm)')
            plt.xlabel('pos X (cm)')
            
            plt.figure("y view")
            # plt.plot(posy, cly, '.', markersize=2)
            for i in range(run_i, run_f):
                dots = plt.plot(posy[i].T, cly[i].T, 'o')
                color = dots[0].get_color()
                if new_pary[i,0] == new_pary[i,1]:
                    plt.plot([new_pary[i,0]*cly[i,0]+new_pary[i,2], new_pary[i,0]*cly[i,-1]+new_pary[i,2]], [cly[i,0], cly[i,-1]], linestyle='--', color=color)
                else:
                    x1, x2 = cly[i, :int(new_pary[i,5])+1], cly[i, int(new_pary[i,5]):]
                    plt.plot([new_pary[i,0]*x1[0]+new_pary[i,2], new_pary[i,0]*x1[-1]+new_pary[i,2]], [x1[0], x1[-1]], linestyle='--', color=color)
                    plt.plot([new_pary[i,1]*x2[0]+new_pary[i,3], new_pary[i,1]*x2[-1]+new_pary[i,3]], [x2[0], x2[-1]], linestyle='--', color=color)           
            plt.ylabel('h (cm)')
            plt.xlabel('pos Y (cm)')
            plt.show()
        
        if plot_distr:
            trk2 =trk.Define(
                "mx", "Numba::select_col(fitX, 0)"
                ).Define(
                "my", "Numba::select_col(fitY, 0)"
                ).AsNumpy({"mx","my", "sigmaX", "sigmaY"})     
            # new_trk2 = new_trk.Define(
            #     "new_sigma", "Numba::select_col(new_fit, 2)"
            #     ).Define(
            #     "new_m", "Numba::select_col(new_fit, 0)"
            #     ).AsNumpy({"new_m", "new_sigma"})   
            double_trk2 = double_trk.Define(
                "double_sigmaX", "Numba::select_col(double_fitX, 4)"
                ).Define(
                "double_sigmaY", "Numba::select_col(double_fitY, 4)"
                ).Define(
                "tx", "Numba::select_col(double_fitX, 1)"
                ).Define(
                "ty", "Numba::select_col(double_fitY, 1)"
                ).Define(
                "Xindex", "Numba::select_col(double_fitX, 5)"
                ).Define(
                "Yindex", "Numba::select_col(double_fitY, 5)"
                ).Filter(index_filter).AsNumpy({"tx", "ty", "double_sigmaX", "double_sigmaY"})
                
            mx = [np.array(v) for v in trk2["mx"]]
            my = [np.array(v) for v in trk2["my"]]
            sigmax = [np.array(v) for v in trk2["sigmaX"]]
            sigmay = [np.array(v) for v in trk2["sigmaY"]]
            # new_m = [np.array(v) for v in new_trk2["new_m"]]
            # new_sigma = [np.array(v) for v in new_trk2["new_sigma"]]
            double_mx = [np.array(v) for v in double_trk2["tx"]]
            double_my = [np.array(v) for v in double_trk2["ty"]]
            double_sigmax = [np.array(v) for v in double_trk2["double_sigmaX"]]
            double_sigmay = [np.array(v) for v in double_trk2["double_sigmaY"]]
            mx = np.array(mx)
            my = np.array(my)
            sigmax = np.array(sigmax)
            sigmay = np.array(sigmay)
            # new_m = np.array(new_m)
            # new_sigma = np.array(new_sigma)
            double_mx = np.array(double_mx)
            double_my = np.array(double_my)
            double_sigmax = np.array(double_sigmax)
            double_sigmay = np.array(double_sigmay)
            
            plt.figure("sigma x")
            bin_sigma1 = 200
            # bin_sigma2 = int(bin_sigma1 * (np.max(new_sigma)-np.min(new_sigma))/(np.max(sigma)-np.min(sigma)))
            bin_sigma3 = int(bin_sigma1 * (np.max(double_sigmax)-np.min(double_sigmax))/(np.max(sigmax)-np.min(sigmax)))
            plt.hist(sigmax, bin_sigma1, histtype='step', label='old')
            # plt.hist(new_sigma, bin_sigma2, histtype='step', label='new')
            plt.hist(double_sigmax, bin_sigma3, histtype='step', label='new')
            plt.yscale('log')
            plt.xlabel('sigma (cm)')
            plt.legend()
            
            plt.figure("sigma y")
            bin_sigma1 = 200
            # bin_sigma2 = int(bin_sigma1 * (np.max(new_sigma)-np.min(new_sigma))/(np.max(sigma)-np.min(sigma)))
            bin_sigma3 = int(bin_sigma1 * (np.max(double_sigmay)-np.min(double_sigmay))/(np.max(sigmay)-np.min(sigmay)))
            plt.hist(sigmay, bin_sigma1, histtype='step', label='old')
            # plt.hist(new_sigma, bin_sigma2, histtype='step', label='new')
            plt.hist(double_sigmay, bin_sigma3, histtype='step', label='new')
            plt.yscale('log')
            plt.xlabel('sigma (cm)')
            plt.legend()
            
            plt.figure("X tangent")
            bin_tan1 = 200
            # bin_tan2 = int(bin_tan1 * (np.max(new_m)-np.min(new_m))/(np.max(m)-np.min(m)))
            bin_tan3 = int(bin_tan1 * (np.max(double_mx)-np.min(double_mx))/(np.max(mx)-np.min(mx)))
            plt.hist(mx, bin_tan1, histtype='step', label='old')
            # plt.hist(new_m, bin_tan2, histtype='step', label='new')
            plt.hist(double_mx, bin_tan3, histtype='step', label='new')
            plt.yscale('log')
            plt.xlabel('tan XZ')
            plt.legend()
            
            plt.figure("Y tangent")
            bin_tan1 = 200
            # bin_tan2 = int(bin_tan1 * (np.max(new_m)-np.min(new_m))/(np.max(m)-np.min(m)))
            bin_tan3 = int(bin_tan1 * (np.max(double_my)-np.min(double_my))/(np.max(my)-np.min(my)))
            plt.hist(my, bin_tan1, histtype='step', label='old')
            # plt.hist(new_m, bin_tan2, histtype='step', label='new')
            plt.hist(double_my, bin_tan3, histtype='step', label='new')
            plt.yscale('log')
            plt.xlabel('tan YZ')
            plt.legend()
            plt.show()