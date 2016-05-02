import numpy as np
import math
import sys
#import ty_any
#import ty_isnumeric
import inspect
import time as tm
#from EMAN2 import EMNumPy
from scipy import io, linalg
import scipy.special as s
#from sympy.mpmath import besselj, bessely, legendre
#from structures import *
from numpy import linalg as LA
from math import pi, sqrt, sin, cos, acos, atan2, factorial
from virus_obj import virus_obj, clnp
#from EMAN2 import EMData

import funcs
from funcs import xyztosph_vec, plgndr, rd_b, euler2R, setL_nord

def find_lmax(l):
	curmax = l[0]
	i = 1
	while i < len(l):
		if (l[i] > curmax):
			curmax = l[i]
		i = i + 1
	return curmax

def set_Q_noise_solve(rule, vk, y, virusobjs, p_theta_eta, Na):
    t_Q=tm.time()

    WEIGHT_INDEX = 5 # the index of the abscissa weights in the rule data structure

    vo = virusobjs # use a shorter name inside the function

    """ initialize variables """

    Ny = y.shape[0]
    Nv = y.shape[1]
    Nzi = rule.shape[0]
    Nc = vo[0].cbar.shape[0]

    """ compute the integral $$Q$$ Eq. (178) its gradient and Hessian """
    Qem = 0
    A=0
    B=0

    # compute the expectation with numerical integrations
    eta=0

    nu=vo[eta].nu
    # unneeded since the lndet_fast function is commented out detV=prod(nu);
    # read tilde b
    b_fn = 'callti.out.read_by_C' # file contains \tilde{b} values computed from mathematica
#    lmax = np.max(vo[eta].clnp.l)
    lmax = find_lmax(vo[eta].clnp.l)
    print lmax
    tilde_b = rd_b(b_fn, lmax)

    for n in xrange(0,Nzi):
        Rabc = euler2R(rule[n, 0:3])
        L=setL_nord(Rabc, vo[eta].clnp.l, vo[eta].clnp.n, vk, vo[eta].Htable, vo[eta].map_unique2lp,tilde_b)
        """ Use fast lndet algorithm """
        Lc = np.dot(L,vo[eta].cbar) # \mu = L*c;
        y_Lc =  y - Lc
        wri = np.dot(rule[n,WEIGHT_INDEX],p_theta_eta[eta][0][:,n])

        A=A+np.sum(wri)
        B=B+np.dot(np.sum(y_Lc**2,axis=0),wri)

    A=A*Na*Na
    noise_var=B/A

    sys.stdout.write('set_Q_noise_solve time: %d\n'%(tm.time()-t_Q))
    return noise_var

    # copied from funcs for debugging and modification
    # omitted here for simplicity
    # returns a matrix L
    pass

def convert_virusobjs(vobj):
	new_vobj = []
	for i in range(0, len(vobj)):
		new_vobj.append(virus_obj())
		tmp = vobj[i][0][0][0]
		new_vobj[i].cbar = tmp['cbar']
		new_vobj[i].clnp_fn = tmp['clnp_fn']
		new_vobj[i].nu_fn = tmp['nu_fn']
		new_vobj[i].q_fn = tmp['q_fn']
		new_vobj[i].BasisFunctionType = tmp['BasisFunctionType']
		new_vobj[i].R1 = tmp['R1']
		new_vobj[i].R2 = tmp['R2']
		new_vobj[i].nu = tmp['nu']
		new_vobj[i].q = tmp['q']
		new_vobj[i].unique_lp = tmp['unique_lp']
		new_vobj[i].map_unique2lp = tmp['map_unique2lp']
		new_vobj[i].map_lp2unique = tmp['map_lp2unique']
		new_vobj[i].Htable = tmp['Htable']
		new_vobj[i].clnp = clnp()
		new_vobj[i].clnp.c = tmp['clnp']['c']
		new_vobj[i].clnp.l = tmp['clnp']['il'][0][0]
		new_vobj[i].clnp.n = tmp['clnp']['in'][0][0]
		new_vobj[i].clnp.p = tmp['clnp']['ip'][0][0]
		new_vobj[i].clnp.optflag = tmp['clnp']['optflag']
	return new_vobj
	pass


for cnt in range(1, 5):
    mat = io.loadmat('set_Q_noise_solve_test%d.mat' % cnt)
    
    # load inputs from matrices in .mat file
    rule = mat['rule']
    vk = mat['vk']
    y = mat['y']
    virusobjs = convert_virusobjs(mat['virusobjs'])
    p_theta_eta = mat['p_theta_eta']
    Na = mat['Na']
    # get the results in its Python counterpart
    noise_var = set_Q_noise_solve(rule, vk, y, virusobjs, p_theta_eta, Na)


    # threshold testing
    if abs(noise_var - mat['noise_var']) > 1e-7:
           print "error out of bound"
           break
