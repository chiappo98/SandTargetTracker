from matplotlib import pyplot as plt
import ROOT
import numpy as np
import glob
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from numpy.linalg import norm

class Waveforms:
    def __init__(self, _channel, _chargeThr):
        self.path = '/mnt/e/data/drift_chamber/'
        self.channel = _channel
        self.chargeThr = _chargeThr
    def import2RDF(self):
        chain = ROOT.TChain("EvFromDig")
        for f in sorted(glob.glob(self.path + 'run_*_0x17520000v1751.root')):
            chain.Add(f)
        df = ROOT.RDataFrame(chain)
        self.dfBranches = df.AsNumpy({"id", "ch00", "ch03"})
    
    
class Tracks:
    def __init__(self, _fname, _channel, _chargeThr):
        self.path = '/mnt/e/data/drift_chamber/'
        self.fname = _fname
        self.channel = _channel
        self.chargeThr = _chargeThr        
    def import2RDF(self):
        df = ROOT.RDataFrame("tree", self.path + self.fname)
        dfBranches = df.Filter(self.channel + ">" + str(self.chargeThr)).AsNumpy({"x", "y", "dc0_c", "dc1_c", "dc2_c", "dc3_c", "dc4_c", "dc5_c"})
        leafnames = ["x", "y", "dc0_c", "dc1_c", "dc2_c", "dc3_c", "dc4_c", "dc5_c"]
        self.trackList = []
        for leaf in leafnames:
            self.trackList.append([np.array(v) for v in dfBranches[leaf]])
    def rotate(self, angle):
        x, y = self.trackList[0], self.trackList[1]
        cord = np.transpose(np.dstack((x,y)).reshape(-1,2))
        angle = angle * np.pi/180
        R = [[np.cos(angle), np.sin(angle)],[-np.sin(angle),np.cos(angle)]]
        new_cord = np.transpose(np.matmul(R,cord))
        self.rotated_x = new_cord[:,0]
        self.rotated_y = new_cord[:,1]
    def Gauss(self, x, A, x0, sigma):
        y = A*np.exp(-(x-x0)**2/(2*sigma**2))
        return y
    def build_profile(self, plot=False):
        fig = plt.figure()
        h2d = plt.hist2d(self.rotated_x, self.rotated_y, 240) #240 bins => 0.25cm/bin
        self.px_at_Theta = np.sum(h2d[0], axis=1)
        self.py_at_Theta = np.sum(h2d[0], axis=0)    
        self.posx_at_Theta = np.linspace(h2d[1][0] + (h2d[1][1]-h2d[1][0])/2, h2d[1][-2] + (h2d[1][-1]-h2d[1][-2])/2, len(self.px_at_Theta))
        self.posy_at_Theta = np.linspace(h2d[2][0] + (h2d[2][1]-h2d[2][0])/2, h2d[2][-2] + (h2d[2][-1]-h2d[2][-2])/2, len(self.py_at_Theta))
        self.pyPar_at_Theta, py_cov = curve_fit(self.Gauss, self.posy_at_Theta, self.py_at_Theta)
        plt.close(fig)
        if plot:
            plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('y')

            fig, ax = plt.subplots(1,2)
            ax[0].plot(self.posx_at_Theta, self.px_at_Theta, '.')
            ax[1].plot(self.posy_at_Theta, self.py_at_Theta, '.')
            ax[1].plot(self.posy_at_Theta, self.Gauss(self.posy_at_Theta, self.pyPar_at_Theta[0], self.pyPar_at_Theta[1], self.pyPar_at_Theta[2]), '-', label='sigma=%.3F' % self.pyPar_at_Theta[2])
            ax[0].set_xlabel('x profile')
            ax[1].set_xlabel('y profile')
            plt.legend()
            plt.show()
    def pca(self, plot=False):
        hit_x = self.trackList[0] - np.mean(self.trackList[0])
        hit_y = self.trackList[1] - np.mean(self.trackList[1])
        hits_pos = np.array([hit_x, hit_y])
        _pca = PCA(n_components = 2).fit(np.transpose(hits_pos))
        self.comp1 = np.array([_pca.components_[0][0], _pca.components_[0][1]])
        self.comp2 = np.array([_pca.components_[1][0], _pca.components_[1][1]])
        weight1, weight2 = _pca.explained_variance_[0], _pca.explained_variance_[1]
        self.pca_angle = np.arccos(np.dot(self.comp1, np.array([1,0]))/(norm(self.comp1)))*180/np.pi
        if (((_pca.components_[0][0]>0)&(_pca.components_[0][1]<0))|((_pca.components_[0][0]<0)&(_pca.components_[0][1]<0))):
            self.pca_angle = -self.pca_angle
        self.centroid = np.array([0, _pca.components_[0][0]])
        
        if plot:
            plt.plot(hits_pos[0,:], hits_pos[1,:], '.', markersize=2)
            for i, (comp, var) in enumerate(zip(_pca.components_, _pca.explained_variance_)):
                plt.plot([0, comp[0]], [0, comp[1]], label=f"Component {i}", color=f"C{i + 2}")
            plt.gca().set( title="2-dimensional dataset with principal components", xlabel="first feature", ylabel="second feature")
            plt.legend()
            plt.show()
    