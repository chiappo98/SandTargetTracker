import numpy as np
from numba import jit, njit
import ROOT


@njit
def _coeff_mat(x, deg):

    mat_ = np.ones(shape=(x.shape[0],deg + 1))
    mat_[:, 1] = x

    if deg > 1:
        for n in range(2, deg + 1):
            # here, the pow()-function was turned into multiplication, which gave some speedup for me (up to factor 2 for small degrees, ...
            # ... which is the normal application case)
            mat_[:, n] = mat_[:, n-1] * x

    # evaluation of the norm of each column by means of a loop
    scale_vect = np.empty((deg + 1, ))
    for n in range(0, deg + 1):
        
        # evaluation of the column's norm (stored for later processing)
        col_norm = np.linalg.norm(mat_[:, n])
        scale_vect[n] = col_norm
        
        # scaling the column to unit-length
        mat_[:, n] /= col_norm

    return mat_, scale_vect
    
@njit
def _fit_x(a, b, scales):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    # due to the stabilization, the coefficients have the wrong scale, which is corrected now
    det_ /= scales

    return det_

@njit
def fit_poly(x, y, deg):
    a, scales_ = _coeff_mat(x, deg)
    p = _fit_x(a, y, scales_)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]

@njit
def nb_curve_fit(x,y):
    par = fit_poly(x,y,1)
    if x.size > 2:
        sigma = np.sqrt(np.sum((y - par[0]*x - par[1])**2)/(x.size - 2))
    else:
        sigma = np.sqrt(np.sum((y - par[0]*x - par[1])**2))
    return np.array([par[0], par[1], sigma])

@njit
def double_sigma(x, y, par1, par2, i):
    var1 = np.sum((y[:i+1] - par1[0]*x[:i+1] - par1[1])**2)
    var2 = np.sum((y[i:] - par2[0]*x[i:] - par2[1])**2)
    return np.sqrt((var1+var2)/(x.size - 2))
    
@ROOT.Numba.Declare(['RVec<double>', 'RVec<double>'], 'RVec<double>')
def fit_track(clY_z, pos):
    return nb_curve_fit(clY_z, pos)

@ROOT.Numba.Declare(['RVec<double>', 'int'], 'double')
def select_col(ndarr, n):
    return ndarr[n]

@ROOT.Numba.Declare(['double', 'double', 'double'], 'double')
def project_track(x, m, q):
    return m*x+q

@ROOT.Numba.Declare(['RVec<double>', 'RVec<double>', 'RVec<double>'], 'RVec<double>')
def find_best_fit(clY_z, pos, fit_par):
    idx = np.arange(0,5)
    new_fit_par = fit_par
    excluded_dot = np.nan
    
    if fit_par[2] > 0.2:
        for i in range(5):
            y = np.take(clY_z, np.where(idx!=i)[0])
            p = np.take(pos, np.where(idx!=i)[0])
            tmp_par = nb_curve_fit(y, p)
            if tmp_par[2] < new_fit_par[2]:
                new_fit_par = tmp_par
                excluded_dot = i
        
        if excluded_dot==4:
            new_fit_par = nb_curve_fit(clY_z[3:], pos[3:])
    
    return new_fit_par

@ROOT.Numba.Declare(['RVec<double>', 'RVec<double>', 'RVec<double>'], 'RVec<double>')
def find_best_fit2(clY_z, pos, fit_par):
    idx = np.arange(0,5)
    new_fit_par = np.array([fit_par[0], fit_par[0], fit_par[1], fit_par[1], fit_par[2], 0])
    
    if fit_par[2] > 0.2:
        for i in range(5):
            y = np.take(clY_z, np.where(idx!=i)[0])
            p = np.take(pos, np.where(idx!=i)[0])
            tmp_par = nb_curve_fit(y, p)
            if tmp_par[2]<new_fit_par[4]:
                new_fit_par = np.array([tmp_par[0], tmp_par[0], tmp_par[1], tmp_par[1], tmp_par[2], i])
           
            if i!=4 or i!=0:
                tmp_par1 = nb_curve_fit(clY_z[:i+1], pos[:i+1])
                tmp_par2 = nb_curve_fit(clY_z[i:], pos[i:])
                tmp_sigma = double_sigma(clY_z, pos, tmp_par1, tmp_par2, i)
               
                if tmp_sigma<tmp_par[2] and tmp_sigma<new_fit_par[4]:
                    new_fit_par = np.array([tmp_par1[0], tmp_par2[0], tmp_par1[1], tmp_par2[1], tmp_sigma, i])
    
    return new_fit_par