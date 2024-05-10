import numpy as np
from matplotlib import pyplot as plt

def drift_t(x, m, v0):
    x_um = x*1e4
    v = v0 - m*x_um # v in units um/ns
    return x_um/v #t in ns

def drift_t_kloe(x, n, a, b):
    return n*np.arctan(a*x**2 + b*x)

if __name__ == "__main__":
    
    nevents = int(1e4)
    tx = np.random.triangular(-0.2, 0, 0.2, nevents)#uniform(-0.2, 0.2, nevents) #normal #
    ty = np.random.triangular(-0.2, 0, 0.2, nevents)#uniform(-0.2, 0.2, nevents)
    px = np.random.uniform(-10, 10, nevents)
    py = np.random.uniform(-3, 3, nevents)
    
    Wvec = (np.array([1, 0, 0]).reshape(3,1) *np.ones((1, nevents)) ).T
    Tvec = (np.array([tx, ty, np.ones(nevents)])).T
    Wpoint = np.zeros((nevents, 3)).T
    Tpoint = np.array([px, py, np.zeros(nevents)])
    
    top_position = Tpoint + 0.5*Tvec.T
    bottom_position = Tpoint - 0.5*Tvec.T
    cell_filter = (top_position[1,:] < 1) & (top_position[1,:] > -1) & (bottom_position[1,:] < 1) & (bottom_position[1,:] > -1)
    
    direction3d = np.cross(Wvec, Tvec).T
    direction3d/=np.linalg.norm(direction3d, axis=0)
    distance = np.abs(np.diag(np.matmul((Tpoint-Wpoint).T, direction3d)))[cell_filter]
    
    plt.figure()
    plt.plot(distance, drift_t(distance, 1.7e-3, 40), '.', markersize=2, label='linear')
    plt.plot(distance, drift_t_kloe(distance, 500, 0.5, 1), '.', markersize=2, label='kloe')
    plt.xlabel("distance (cm)")
    plt.ylabel("time (ns)")
    plt.legend()
    
    plt.figure()
    plt.plot(distance, distance*1e4/drift_t(distance, 1.7e-3, 40), '.', markersize=2, label='linear')
    plt.plot(distance, distance*1e4/drift_t_kloe(distance, 500, 0.5, 1), '.', markersize=2, label='kloe')
    plt.xlabel("distance (cm)")
    plt.ylabel("velocity (um/ns)")
    plt.legend()
    
    # plt.figure()
    # plt.hist(distance, 200)
    # plt.xlabel('distance (cm)')
    
    plt.figure()
    plt.hist(drift_t(distance, 1.7e-3, 40),100, histtype='step', label='linear')
    plt.hist(drift_t_kloe(distance, 500, 1.5, 0.5), 100, histtype='step', label='kloe')
    plt.xlabel('drift time (ns)')
    plt.legend()
    
    # plt.figure()
    # plt.plot(Tpoint[0,:][cell_filter][distance<0.3], Tpoint[1,:][cell_filter][distance<0.3], '.', markersize=3)
    # plt.plot(Tpoint[0,:][cell_filter][(distance>=0.3)&(distance<0.7)], Tpoint[1,:][cell_filter][(distance>=0.3)&(distance<0.7)], '.', markersize=3)
    # plt.plot(Tpoint[0,:][cell_filter][(distance>=0.7)&(distance<1)], Tpoint[1,:][cell_filter][(distance>=0.7)&(distance<1)], '.', markersize=3)
    # plt.plot(Tpoint[0,:][cell_filter][distance>=1], Tpoint[1,:][cell_filter][distance>=1], '.', markersize=3)
    # plt.axhline(0, -10, 10)
    
    plt.show()