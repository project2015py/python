import pp
import numpy as np
import math
import sys
import scipy.linalg
# import ty_any
#import ty_isnumeric
import inspect
import time as tm
import funcs
from funcs import * # rd_b, euler2R, setL_nord, lndet_fast, xyztosph_vec, set_Ttable_nord, plgndr
import structures
from structures import Clnp, SingleVirus
#from EMAN2 import EMNumPy
from scipy import io
import scipy.special as s
#from sympy.mpmath import besselj, bessely, legendre
#from structures import *
from numpy import linalg as LA
from math import pi, sqrt, sin, cos, acos, atan2, factorial
#from EMAN2 import EMData


def set_p_theta_eta_loglike_diagV(rule, vk, y, noise_var, virusobjs):
    """
    compute $$p(\theta_i,\eta_i|y_i,\bar{c}, V, q)$$ by Eq. (25) and the log
    likelihood $$p(y|\bar{c}, V, q)$$ by Eq. (320)

    Input parameters:
      rule: integration rule object
      Lall: precomputed data structures, please set_Lall()
      y: image data
      Qnoise: covariance matrix of the noise
      cbar, V, q: current estimation of mean $$\bar{c}$$, variance $$V$ and
      prior class probability $$q$, respectively.

    Output parameters:
      p: $$p(\theta_i,\eta_i|y_i,\bar{c}, V, q)$$ from Eq. (25)

      This is the diagonal V version of set_p_theta_eta
      Use fast algorithm to compute lndet(Sigma) and Sigma_inverse
      read tilde_b once
    """
    t_p = tm.time()

    WEIGHT_INDEX = 5  # the index of the abscissa weights in the rule data structure

    vo = virusobjs  # use a shorter name inside the function

    """ initialize variables """
    Neta = len(vo)
    # Ny = size(y, 1);
    Ni = y.shape[1]
    Nzi = rule.shape[0]
    exponents = np.zeros((Neta, 1), dtype=object)

    """ compute the scaling factor of the exponents """

    for eta in xrange(0, Neta):

        print "Virus object %d / %d" % (eta, Neta - 1)

        Nc = vo[eta].nu.shape[0]
        exponents_eta = np.zeros((Ni, Nzi))

        lndetV = np.sum(np.log(vo[eta].nu))

        # read tilde b
        b_fn = 'callti.out.read_by_C'  # file contains \tilde{b} values computed from mathematica
        lmax = np.max(vo[eta].clnp.l)
        tilde_b = rd_b(b_fn, lmax)

        """ATTENTION
        I change parfor to for
        """
        ppservers = ()
        ncpus = 8
        job_server = pp.Server(ncpus, ppservers=ppservers)
        print "Starting pp with", job_server.get_ncpus(), "workers"

        jobs = []
        L = []

        for n in xrange(0, Nzi):
            Rabc = euler2R(rule[n, 0:3])
            func_name = "setL_nord"
            args_ls = "(Rabc, vo[eta].clnp.l, vo[eta].clnp.n, vk, vo[eta].Htable, vo[eta].map_unique2lp, tilde_b, )"
            depfuncs_ls = "(plgndr, set_Ttable_nord, xyztosph_vec, )"
            modules_ls = '("math", "funcs", "numpy as np", "scipy", )'
            exec ("jobs.append(job_server.submit(%s, args = %s, depfuncs = %s, modules = %s))" % (
                func_name, args_ls, depfuncs_ls, modules_ls))

        for n in xrange(0, Nzi):
            L = jobs[n]()
            lndet_Sigma = lndet_fast(noise_var, L, vo[eta].nu,
                                     lndetV)  # this function has a relatively small difference with matlab, this is where the difference comes
            Lc = np.dot(L, vo[eta].cbar)

            y_Lc = y - Lc
            y_Lc_Sigma = y_Lc_Sigma_fast(noise_var, L, vo[eta].nu, y_Lc)  # use fast algorithm
            assert np.all(
                np.isfinite(y_Lc_Sigma)), 'set_p_theta_eta_loglike_diagV: assert(all(isfinite(y_Lc_Sigma))): eta %d n %d\n' % (
                eta, n)

            exponents_eta[:, n] = -0.5 * (y_Lc_Sigma[:] + lndet_Sigma)

        job_server.print_stats()

        exponents[eta][0] = exponents_eta
        if not np.all(np.isreal(exponents[eta][0])):
            sys.exit('The exponents matrix is not real! eta %d\n' % eta)

        if eta == 0:
            maxexp = exponents[eta][0].max(1).reshape(-1, 1)  # column vector of size Ni
            minexp = exponents[eta][0].min(1).reshape(-1, 1)  # column vector of size Ni
        else:
            maxexp = np.maximum(maxexp, exponents[eta][0].max(1).reshape(-1, 1))  # column vector of size Ni
            minexp = np.minimum(minexp, exponents[eta][0].min(1).reshape(-1, 1))  # column vector of size Ni

    """
        compute the scaling factor for the exponents in order to prevent overflow
        maxexp and minexp are column vectors with size Ni.
        """
    if maxexp.shape[0] != Ni or minexp.shape[0] != Ni:
        sys.exit('size of maxexp %d, size of minexp %d, shhould equal to Ni %d!\n' % (maxexp.shape[0], minexp.shape[0], Ni))

    expshift = np.zeros((Ni, 1))  # vector of size Ni
    for img in xrange(0, Ni):
        expshift[img][0] = scaling_rule(minexp[img][0], maxexp[img][0])

    """ compute $$p(\theta_i,\eta_i | y_i, \bar{c}, V, q}$$ """
    sys.stdout.write('computing p...\n')
    Zeta = np.zeros((Neta, 1), dtype=object)  # numerator $$p(y_i|\theta_i,\eta_i,\bar{c}^{\eta_i},V_{\eta_i})q_{\eta_i}$$
    Zetabar = np.zeros((Ni, 1))  # denominator
    p = np.zeros((Neta, 1), dtype=object)

    for eta in xrange(0, Neta):
        Zeta[eta][0] = np.zeros((Ni, Nzi))
        for n in xrange(0, Nzi):
            Zeta[eta][0][:, n] = np.exp(exponents[eta][0][:, n] + expshift[:].flat) * vo[eta].q
            Zetabar[:] = Zetabar[:] + (Zeta[eta][0][:, n] * rule[n][WEIGHT_INDEX]).reshape(-1, 1)

    assert np.all(Zetabar[:] != 0.0), 'set_p_theta_eta_loglike_diagV: assert(all(Zetabar(:)~=0.0))'

    for eta in xrange(0, Neta):
        p[eta][0] = Zeta[eta][0] / Zetabar

    """ compute the log likelihood """
    sys.stdout.write('computing the log likelihood...\n')
    loglike = np.sum(np.log(Zetabar)) - np.sum(expshift)  # need to undo the effect of shifting exponents
    # fprintf(1, 'set_p_theta_eta_nu_noL: log likelihood %.16g.\n', loglike);

    sys.stdout.write('set_p_theta_eta_loglike_diagV time: %d\n' % (tm.time() - t_p))

    print "Total runtime: ", tm.time() - t_p

    return p, loglike


def vobj_mat2py(mat_vobj):
    vobj_attrs = ['clnp_fn', 'nu_fn', 'q_fn', 'cbar', 'nu', 'unique_lp', 'map_lp2unique', 'map_unique2lp', 'Htable']
    vobj_attrs_num = ['BasisFunctionType', 'R1', 'R2', 'q']
    clnp_attrs = ['c', 'optflag']
    clnp_attrs_i = ['l', 'n', 'p']

    py_vobj = []
    for obj in mat_vobj:
        mat_tmp = obj[0][0][0]

        py_tmp = SingleVirus()
        for attr in vobj_attrs:
            exec ("py_tmp.%s = mat_tmp['%s']" % (attr, attr))
        for attr in vobj_attrs_num:
            exec ("py_tmp.%s = mat_tmp['%s'][0][0]" % (attr, attr))

        py_tmp.clnp = Clnp()
        for attr in clnp_attrs:
            exec ("py_tmp.clnp.%s = mat_tmp['clnp']['%s'][0][0]" % (attr, attr))
        for attr in clnp_attrs_i:
            exec ("py_tmp.clnp.%s = mat_tmp['clnp']['i%s'][0][0]" % (attr, attr))

        py_vobj.append(py_tmp)

    return py_vobj


for cnt in range(1, 2):
    mat = io.loadmat('set_p_theta_eta_loglike_diagV.mat')

    # load inputs from matrices in .mat file
    rule = mat['rule']
    vk = mat['vk']
    y = mat['y']
    virusobjs = vobj_mat2py(mat['virusobjs'])
    noise_var = mat['noise_var']
    # get the results in its Python counterpart

    (p, loglike) = set_p_theta_eta_loglike_diagV(rule, vk, y, noise_var, virusobjs)

    # threshold testing
    #if abs(noise_var - mat['']) > 1e-7:
    #      print "error out of bound"
    #     break
