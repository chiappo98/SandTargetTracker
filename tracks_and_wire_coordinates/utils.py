import numpy as np
from numba import njit
import ROOT


@njit
def interpole_m(x,y):
    return (y[-1]-y[0])/(x[-1]-x[0])

@njit
def interpole_q(x,y,m):
    return y[0]-m*x[0]

@ROOT.Numba.Declare(['RVec<double>', 'RVec<double>'], 'double')
def fit_m(clY_z, pos):
    return interpole_m(clY_z, pos)

@ROOT.Numba.Declare(['RVec<double>', 'RVec<double>', 'double'], 'double')
def fit_q(clY_z, pos, m_fit):
    return interpole_q(clY_z, pos, m_fit)

@ROOT.Numba.Declare(['RVec<double>', 'RVec<double>', 'double', 'double'], 'double')
def sigma_fit(clY_z, pos, m_fit, q_fit):
    return np.sqrt(np.sum((pos - m_fit*clY_z - q_fit)**2)/(5-2))

@ROOT.Numba.Declare(['RVec<double>', 'RVec<double>', 'double', 'double', 'double'], 'RVec<double>')
def find_best_fit(clY_z, pos, m, q, sigma):
    idx = np.arange(0,5)
    new_sigma = sigma
    new_fit_par = np.array([m, q, sigma, np.nan])
    if sigma > 0.2:
        for i in range(5):
            y = np.take(clY_z, np.where(idx!=i)[0])
            p = np.take(pos, np.where(idx!=i)[0])
            tmp_m = (p[-1]-p[0])/(y[-1]-y[0])
            tmp_q = p[0]-tmp_m*y[0]
            tmp_sigma = np.sqrt(np.sum((p - tmp_m*y - tmp_q)**2)/(4-2))
            if tmp_sigma<new_sigma:
                new_sigma = tmp_sigma
                new_fit_par = np.array([tmp_m, tmp_q, tmp_sigma, i])
        if new_fit_par[3]==4 or new_fit_par[3]==3:
            new_fit_par[0] = (pos[-1]-pos[3])/(clY_z[-1]-clY_z[3])
            new_fit_par[1] = pos[0]-tmp_m*clY_z[0]
            new_fit_par[2] = 0
    return new_fit_par
