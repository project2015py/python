__author__ = 'kevin'



import numpy as np
#import h5py
import time as tm
import scipy as si
import funcs as fun
import scipy
from structures import *


def setHQem(nu):  # rule 2D matrix; vk 2D
    data = si.io.loadmat('fmin.mat', squeeze_me=True, struct_as_record=False)

    t_Q = tm.time()  #clock:matlab function, return the date and time

    WEIGHT_INDEX = 5

    vo = data['vobj']

    Ny = data['y'].shape[0]
    Nv = data['y'].shape[1]
    rule = data['EM_iter'].rule
    vk = data['vk']
    noise_var = data['pixelnoisevar']
    p_theta_eta = data['p_theta_eta']
    y = data['y']
    Nzi = data['EM_iter'].rule.shape[0]
    tilde_b = data['tilde_b']
    eta = data['eta']

    # print (type(vo[0].flags[0]))
    Nc = vo[0].cbar.shape[0]
    #vo, a cell array, each element is a virus object
    #cbar, a vector of double precision, attribute of the object
    #matlab mathwork

    HQem = np.zeros((Nc, Nc), float)  #numpy array

    #eta = 0
    nu = vo[eta].nu
    if len(nu.shape)<2:
        d1 = nu.shape[0]
        nu = si.reshape(nu, [d1, 1])
    lndetV = np.sum(np.log(nu))

    #b_fn = 'callti.out.read_by_C'
    #lmax = vo[eta].clnp.l.max(0)  #il in matlab is l in python, ndarray
    #tilde_b = fun.rd_b(b_fn, lmax)

    for n in range(Nzi):  #n an index for a matrix
        Rabc = fun.euler2R(rule[n, 0:3])  #rule is a matrix
        L = fun.setL_nord(Rabc, vo[eta].clnp.l, vo[eta].clnp.n, vk, vo[eta].Htable, vo[eta].map_unique2lp,
                      tilde_b)  #L is a ndarray
        #clnp.in happens to be a keyword in python, how to fix this.
        lndet_Sigma = fun.lndet_fast(noise_var, L, nu, lndetV)  #ndarray

        Lc = np.dot(L, vo[eta].cbar)  #cbar is a list. Transferred to a ndarray before multiplication. Output is a ndarray.
        y_Lc = (y.transpose()-Lc).transpose()  #np.substract(y, Lc)  Good! Translate bxsfun() in Python
        wri = np.dot(rule[n][WEIGHT_INDEX], p_theta_eta[eta][:, n])  #.*, element-wise; *, dot product in matlab ???No matrix, all ndarrays???
        sum_wri = np.sum(wri)
        D, M = fun.LT_SigmaInv_fast(noise_var, L, vo[eta].nu, y_Lc)  #D, M is ndarray according to translation of Yu
        repwri = np.tile(np.sqrt(wri.T), (Nc, 1))
        D = D * repwri
        D = np.dot(D, D.T)

        y_Lc_Sigma = fun.y_Lc_Sigma_fast(noise_var, L, nu, y_Lc)  #ndarray

        HQem = HQem + (-2 * D + M * sum_wri) * M

    HQem = -HQem
    # sys.stdout.write('set_Q_dV_unc time: %d\n' % (tm.time() - t_Q))

    return HQem