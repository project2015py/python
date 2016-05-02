#!/usr/bin/env python
# funcs.py
# Zhengyu Cai, Xiao Ma, Yunhan Wang
# Oct 23, 2013

""" functions called by hetero, notice:
1) write your name in the function commands so that others can get you if any questions
2) difference between matlab and python:
    use np.dot(A,B) instead of A*B for matrix multiplication
    python starts from 0, while matlab from 1:
        matlab [1:n] -> python [x for x in range(1, n+1)]
        matlab a(n) -> python a[n-1]
    index overflow: python does not allow index out of range
    python have very few primitive methods for matrix,
        so consider Numpy whenever a matrix appears
        (e.g. [1,2]*2 in python result in [1,2,1,2] but rather [2,4])
    array slicing using [] rather than {}
    integer and float : 5/2 returns 2 in Python while we want 2.5!
    different major order between Matlab(column major) and Python(use 'F' when creating ndarray)
        x = [1,0;1,0]
        by default, nonzero(x) returns [0,2] (the value equals Matlab find(x') but not grouped well)

    ndarray only allows certain number types, so it will just accept the real part of a complex number
        if the dtype is declared to be "float"! if the numbers in this array contains complex, please
        set "dtype=fun.complex()" (e.g. if cmd[i].operator==u'realrecip_2DFFT' in hetero.py)
3) APIs are crucial !!! must get clear what are inputs and which return type is needed !
4) TEST each unit codes before adding to this file, test this file before submission
Yunhan Wang 1/2014


Description: A package of all sub-routines as of in Matlab, get parameters from hetero.py.
Updated by Yunhan Wang 8/28/2014

"""

import numpy as np
import math
import helper as h
import hetero as t
import sys
import scipy.linalg
import ty_nnz
import ty_any
import ty_isnumeric
import inspect
import time as tm
#from EMAN2 import EMNumPy
from scipy import io
import scipy.special as s
from mpmath import besselj, bessely, legendre
from structures import *
from numpy import linalg as LA
from math import pi, sqrt, sin, cos, acos, atan2, factorial
#from EMAN2 import EMData
#from mpi4py import MPI
#import pp
import mpi4py as MPI
import scipy.io as sio
import NuFunc as objFun
import NuGrad as grad
import NuHess as hess
from scipy.optimize import minimize

def diary(x) :
    """ Save text of MATLAB session.
    change the stdout to not only print in console, but also write to file x
    __author__ = 'Yunhan Wang' """
    t.log.toFile(x)

def rng(x) :
    """ set the random seed to be x (cmd[i].pseudorandomnumberseed)
    __author__ = 'Yunhan Wang' """
    np.random.seed(x)

def vk_indexset(Na,Nb,deltachia,deltachib):
    """ Determine a minimal subset of 2-D reciprocal space such that
    conjugate symmetry fills in all of reciprocal space.  Assume
    that all images have the same dimensions and extract the
    dimensions from the first of the boxed 2-D real-space images.
    __author__ = 'Yunhan Wang' """
    #  ny4minmalset:  number of point (scaler)
    kx, ky =set_k2D(Na,Nb,deltachia,deltachib)
    ix, iy =set_indices2D(Na,Nb)
    minimalset=get_minimal_set(Na,Nb)
    iiminimalset,vkminimalset,ixminimalset,iyminimalset=extract(minimalset,kx,ky,ix,iy)
    Ny4minimalset=2*np.amax(find_operators(minimalset,1,"=").shape) # ndarray multiplication
    # Compute magnitude of the reciprocal space vector.
    a = vkminimalset[:,0]
    b = vkminimalset[:,1]
    vkmag = np.sqrt(np.power(a,2) + np.power(b,2))
    # Package in a structure
    return [Ny4minimalset, iiminimalset, vkminimalset, ixminimalset, iyminimalset, vkmag]
    #       8282            [1..8236]T     4141,2       4141,1          4141,1

    ##### -> for vk_indexset()
def set_k2D(Na,Nb,deltachia,deltachib): # notice Na as index should be Na-1
    """function [ka,kb]=set_k2D(Na,Nb,deltachia,deltachib)
    deltachia and deltachib are the sampling intervals in the
    real space image.  Using values of 1 gives pixel-size units.
    __author__ = 'Yunhan Wang' """
    k1D_a=set_k1D(Na,deltachia)
    k1D_b=set_k1D(Nb,deltachib)
    ka=np.zeros((Na,Nb), float)
    for i in range(Nb):
        ka[:,i]=k1D_a[:,0]
    kb=np.zeros((Na,Nb), float)
    for i in range(Na):
        kb[i,:]=k1D_b.T
    return np.asarray([ka, kb])

    #### -> for vk_indexset() -> for set_k2D()
def set_k1D(N,deltachi):
    """__author__ = 'Yunhan Wang' """
    k=np.zeros((N,1), float)
    for n in range(N): # index into Matlab matrix
        k[n]=index2k(n+1,N,deltachi) # not n but n+1
    return k

def index2k(n,N,deltachi):
    """ function k=index2k(n,N,deltachi)
    n in 1:N is an index into a Matlab vector/matrix
    __author__ = 'Yunhan Wang' """
    N2=float(N)/2
    if n<1 or n>N:
        h.printf(t.log,'index2k: old n %d N %d'%(n,N))
        n=(n-1)%N + 1
        h.printf(t.log,'index2k: new n %d N %d'%(n,N))

    nnear0=n-1
    if nnear0 >= N2:
        nnear0=nnear0-N
    # units of cycles/(units of deltachi) not radians/(units of deltachi)
    return float(nnear0)/(N*deltachi)
    #### <- for set_k2D() <- for vk_indexset()
def set_indices2D(Na,Nb):
    """function [ia,ib]=set_indices2D(Na,Nb) modeled on
    function [ka,kb]=set_k2D(Na,Nb,deltachia,deltachib)
    __author__ = 'Yunhan Wang' """
    v=np.asarray([x for x in range(1,Na+1)])
    v = v.T
    ia=np.zeros((Na,Nb), float)
    for i in range(Nb):
        ia[:,i]=v
    v=np.asarray([x for x in range(1,Nb+1)])
    ib=np.zeros((Na,Nb), float)
    for i in range(Na):
        ib[i,:]=v
    return np.asarray([ia,ib])

def extract(mset,kx,ky,ix,iy):
    """  __author__ = 'Yunhan Wang' """
    iiset=find(mset)
    vkset=np.zeros((iiset.shape[0],2), float) # not 1
    vkset[:,0]=get(kx,iiset)
    vkset[:,1]=get(ky,iiset)
    tmp_ixset=get(ix,iiset)
    tmp_iyset=get(iy,iiset)
    ixset = np.asarray(tmp_ixset).reshape(-1,1)
    iyset = np.asarray(tmp_iyset).reshape(-1,1)
#     ixset = np.asarray([[x] for x in range(4141)])
#     iyset = np.asarray([[x] for x in range(4141)])
    return [iiset,vkset,ixset,iyset]

def get_minimal_set(ndim1,ndim2):
    """(ndim1,ndim2)=dimension of rectangular matrix,
    the DC sample is at (1,1)
    __author__ = 'Yunhan Wang' """
    minimal_set=np.zeros((ndim1,ndim2), float)
    got_it=np.zeros((ndim1,ndim2), float)
    for x1 in range(ndim1): # index into Matlab matrix
        x2=conj_sym_index_1D(x1,ndim1-1) # index start from 0
        for y1 in range(ndim2): # index into Matlab matrix
            y2=conj_sym_index_1D(y1,ndim2-1)
            if got_it[x2,y2] == 0:  # have I already got the conjugate symmetric point?
            # no, I don't have it
                minimal_set[x1,y1] = 1
                got_it[x1,y1]=1
                got_it[x2,y2]=1
    min_got_it=np.amin(got_it[:])
    max_got_it=np.amax(got_it[:])
    if min_got_it!=1 or max_got_it!=1:
        h.printf(t.log,'get_minimal_set: ndim1 %d ndim2 %d min(got_it(:)) %g max(got_it(:)) %g'
                 %(ndim1,ndim2,min_got_it,max_got_it))
    return minimal_set

    #### -> for vk_indexset() -> for get_minimal_set()
def conj_sym_index_1D(ii,N):
    """ ii is the index to a Matlab vector: ii\in\{1,\dots,N\}
    __author__ = 'Yunhan Wang' """
    if ii<0 or ii>N:
        h.errPrint('ii %d N %d'%(ii,N))
    jj=N+1-ii if ii!=0 else ii
    return jj
    #### <- for get_minimal_set() <- for vk_indexset()

def find_operators(a,val,op):
    """ finds the linear indices of arr that are equal to/less than/greater than val,
    same as find(A=a), find(A<a), find(A>a)
    __author__ = 'Yunhan Wang' """
    arr = a.flatten()
    if op == "=":
        return np.arange(len(arr))[arr == val]
    elif op == "<":
        return np.arange(len(arr))[arr < val]
    elif op == ">":
        return np.arange(len(arr))[arr > val]

def find(arr):
    """ find(X), matlab find is col first then row, while python is row first, so transpose it
    __author__ = 'Yunhan Wang' """
    return np.transpose(np.nonzero(arr.T.flat))
    #### <- for vk_indexset()

def rd_rule(fn_rule):
    """ load the rule data file into a ndarray
    __author__ = 'Yunhan Wang' """
    return np.loadtxt(fn_rule)

def rd_b(b_fn, l):
    """ modified from rd_b.c
    Input:
    b_fn: the file name of tildeb data file:
    lmax: max l

    Example usage:
    tildeb=rd_b('/home/hours/yz247/research/EM4.4.ti.bigger_l.mathematica/callti.out.read_by_C',45)
    Yili Zheng
    Feb. 6, 2008

    Corrected by adding fclose()
    Qiu Wang
    07/01/2010

    Precondition: lmax is integer
    __author__ = 'Yunhan Wang'
    Oct 30, 2013 """
    lmax = int(l)
    nmax= lmax/30
    mmax=lmax/5
    print(lmax)
    tildeb=np.zeros((lmax+1,nmax+1,mmax+1), float)
    data = _read_file(b_fn, lmax)

    for tmp1, tmp2, tmp3, l, n in data:
        rd_l = tmp1
        rd_n = tmp2
        if l!=rd_l or n!=rd_n:
            h.errPrint('rd_b: l %d rd_l %d n %d rd_n %d'%(l,rd_l,n,rd_n))
        if h.slct(l,n)==1:
            nstart, lp = ((l%2+1)%2)*[n,'even'] + (l%2%2)*[n+1, 'false']
            for m in range(nstart, l/5+1):
                try:
                    tildeb[l,n,m]=tmp3[m-nstart]
                except IndexError:
                    h.errPrint('rd_b: l %s, l %d n %d m %d'%(lp,l,n,m))
    return tildeb

    #### -> for rd_b1(b_fn, lmax)
def _read_file(b_fn, lmax):
    """ Precondition: all float that will be scanned in the form of %lg is float type with '.'
    Counterexample: %lg could also scan in 1 besides 1.0 in Matlab code, but here we does not accept it
    __author__ = 'Yunhan Wang'
    Oct 30, 2013 """
    # construct an array, each row is the two indexes of the integers in b_fn(will be replaced by integers), arrays of %lg, l, n
    op, num = [], 0
    for l in range(lmax+1):
        for n in range(l/30+1):
            op.append([num,num+1,[],l,n])
            num = num+2
    try:
        n = -1
        for i in open(b_fn,'r'):
            try: # read integers in the file into the matrix, stop when op says it is enough (enough indexes)
                iarr=map(int, i.split()) # the op[0], op[1] are indexes of the matrix to locate the fscanf(fid,'%d%d',2)
                for j in iarr:
                    if n == num-1:
                        return op # already read in enough data
                    n += 1
                    op[n/2][n%2] = j
            except ValueError:
                try:
                    farr=map(float, i.split())
                    op[n/2][2] += farr
                except ValueError:
                    # the error message is different from Matlab,
                    # here l, n indicates the last group read successfully,
                    # not the first group read failed
                    # so this l,n is the previous group of Matlab's l,n
                    h.errPrint('rd_b: l %d n %d reading l,n then stop'%(op[n/2][3],op[n/2][4]))
        if n < num - 1:
            h.errPrint('rd_b: l %d n %d reading l,n then stop'%(op[n/2][3],op[n/2][4]))
    except IOError, e: # error opening file
        h.errPrint('rd_b: fopenmsg %s opening %s'%(e,b_fn))
    return op
    #### <- for rd_b1(b_fn, lmax)

def get(a, b):
    """ Precondition: b is N*1
    similar to a(b) in Matlab, where a and b are both matrix,
    return a flattened array, since it is assigned to x[:,1] which is degenerated from N*1 to N
    __author__ = 'Yunhan Wang' """
    return np.reshape(a, np.product(a.shape), 'F')[b].flat

####### take care of these below #########
def virusobj_read(fn_clnp, fn_nu, fn_q):
    """ Return: numpy ndarray (matlab cell)
    __author__ = 'Xiao Ma'"""
#    Neta=len(fn_clnp);
    h.printf(t.log,'length of fn_clnp: %d'%len(fn_clnp))
    Neta = len(fn_clnp)
    if len(fn_nu)!=Neta or len(fn_q)!=Neta:
        h.errPrint('virusobj: one or more of fn_nu and fn_q are not Neta %d long\n'%(Neta))
    if Neta==0:
        h.printf(t.log,'virusobj_read: Neta = 0')
        return
#    h.errPrint('fn_clnp: %d'%(len(fn_clnp))+'  fn_nu: %d'%(len(fn_nu))+'  fn_q: %d'%(len(fn_q)))
#    vobj=np.ndarray(shape=(Neta,1), dtype=float)
    vobj=[]
    for eta in range(Neta):
        vobj.append([])
        vobj[eta]=SingleVirus()
        vobj[eta].clnp_fn = fn_clnp[eta]
        vobj[eta].nu_fn = fn_nu[eta]
        vobj[eta].q_fn = fn_q[eta]
        header1, header2, vobj[eta].clnp = rd_clnp(vobj[eta].clnp_fn)
        vobj[eta].cbar = vobj[eta].clnp.c
        vobj[eta].BasisFunctionType = header1
        vobj[eta].R1 = header2[0]
        vobj[eta].R2 = header2[1]
        h.printf(t.log,'virusobj_read: eta %d, R1 %g, R2 %g'%(eta,header2[0],header2[1]))
        indices=[vobj[eta].clnp.l, vobj[eta].clnp.n, vobj[eta].clnp.p]
        uniqueindices=unique([indices,'rows'])
        if len(indices[0])!=len(uniqueindices[0]):
            h.errPrint('virusobj_read: repeated weights')
        del indices, uniqueindices
        (vobj[eta].nu,count)=rd_nu(vobj[eta].nu_fn)
        print '+####################+'
        if count!=0 and count!=len(vobj[eta].cbar):
            h.errPrint('virusobj_read: lengths of nu and cbar are different')

        (vobj[eta].q,count)=rd_q(vobj[eta].q_fn)

    tmp=0
    for eta in range(Neta):
        tmp = tmp+vobj[eta].q
    if abs(tmp-1.0)>float(1.0e-12):
        h.errPrint("virusobj_read: vobj{eta}.q summed over eta does not equal 1")
    return vobj


def rd_clnp(fn):
    """ __author__ = 'Xiao Ma' """

    arr=[]
    with open(fn, 'r') as f:
        for line in f:
            arr.append(line.split())
    c = np.asarray(arr)
    temp = np.asarray(arr[2:len(c)])
    temp = temp.astype(np.float)
    clnp = Clnp()
    clnp.l = temp[:,0].astype(np.float).T
    clnp.n = temp[:,1].astype(np.float).T
    clnp.p = temp[:,2].astype(np.float).T
    clnp.optflag = temp[:,3].astype(np.float).T
    clnp.c = temp[:,4].astype(np.float).T
    header1 = [float(c[0][0])]
    header2 = map(float,c[1])
    """
    header1=[1.000]
    header2=[-1.000,197.4000]
    clnp = Clnp()
    clnp.set_l(np.asarray([0, 0, 0, 0, 0,
     6, 6, 6, 6, 6,
    10, 10, 10, 10, 10]))
    clnp.set_n(np.asarray([0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
    0, 0, 0, 0, 0]))
    clnp.set_p(np.asarray([1, 2, 3, 4, 5,
     1, 2, 3, 4, 5,
    1, 2, 3, 4, 5]))
    clnp.set_optflag(np.asarray([1, 1, 1, 1, 1,
     1, 1, 1, 1, 1,
    1, 1, 1, 1, 1]))
    if eta==0:
        clnp.set_optflag(np.asarray([-107.6880, -33.3940, 0.9237, 14.7836,
                                     22.7498, -11.8110, -15.0203, -3.2247,
                                     4.7884, 4.0418, 12.7056, 2.2913,
                                     -1.8318, -2.4134, -3.3936]))
    else:
        clnp.set_optflag = np.asarray([-104, -34, 1, 14, 23, -13, -15, -4, 6, 4, 11, 2, -1, -4, -3])
    """
    return [header1, header2, clnp]

def unique(varargin):
    """ __author__ = 'Xiao Ma' """
    varargout = np.ndarray(shape=(len(varargin[0][0]),3), dtype=float)
    varargout = np.transpose(np.asarray(varargin[0]))
    #print varargout
    ''''
    varargout = np.ndarray(shape=(15,3), dtype=float)
    varargout[0]=[0, 0, 1]
    varargout[1]=[0, 0, 2]
    varargout[2]=[0, 0, 3]
    varargout[3]=[0, 0, 4]
    varargout[4]=[0, 0, 5]
    varargout[5]=[6, 0, 1]
    varargout[6]=[6, 0, 2]
    varargout[7]=[6, 0, 3]
    varargout[8]=[6, 0, 4]
    varargout[9]=[6, 0, 5]
    varargout[10]=[10, 0, 1]
    varargout[11]=[10, 0, 2]
    varargout[12]=[10, 0, 3]
    varargout[13]=[10, 0, 4]
    varargout[14]=[10, 0, 5]
    '''
    return varargout
'''
def uniquelegacy(a, options):
    """ __author__ = 'Xiao Ma' """
    return [b,ndx,pos]

def uniqueR2012a(a,options):
    """ __author__ = 'Xiao Ma' """
    return [c,indA,indC]
'''
def rd_nu(fn):
    """ __author__ = 'Xiao Ma' """
    """
    #nu = np.ndarray(shape=(15,1), dtype=float)
    arr=[]
    with open(fn, 'r') as f:
        for line in f:
            arr.append(line.split())
    nu = np.asarray(arr).astype(np.float).T
    count = len(nu)
    for i in range(count):
        nu[i] = float(nu[i])
    '''
    if eta == 1:
        nu=np.asarray([2.1538, 0.6679, 0.2692, 0.2957, 0.4550, 0.2692, 0.3004, 0.2692, 0.2692, 0.2692, 0.2692, 0.2692, 0.2692, 0.2692, 0.2692])
    else:
        nu=np.asarray([2.0800, 0.6800, 0.2600, 0.2800, 0.4600, 0.2600, 0.3000, 0.2600, 0.2600, 0.2600, 0.2600, 0.2600, 0.2600, 0.2600, 0.2600])
    count = 15
    '''
    """
    with open(fn) as f:
        nu = np.loadtxt(f)
        count = nu.size
        if count==0:
            print 'rd_nu: nu has zero length.  Therefore will solve a homogeneous problem.'


    ii=(nu<0.0).nonzero()[0]
    if ii.size>0:
        sys.exit('rd_nu: %d negative elements in nu\n'%ii.size)

    return nu,count

def rd_q(fn):
    """ __author__ = 'Xiao Ma' """
    """
    arr=[]
    with open(fn, 'r') as f:
        for line in f:
            arr.append(line.split())
    q = float(arr[0][0])
    count = 1
    '''
    if eta==0:
        q=0.6000
    else:
        q=0.4000
    count=1
    '''
    return [q, count]
    """
    q = -1
    with open(fn) as f:
        q = float(f.readline())

    if q<0:
        sys.exit('rd_q: q %g < 0\n'%q)

    return q,1

#  the same function as the previous one
#vobj: virusobj
#vk(1:Ny,1:2): list of reciprocal-space image frequency vectors
#adds the following fields to vobj:
#vobj{eta}.unique_lp, vobj{eta}.map_lp2unique, vobj{eta}.map_unique2lp, vobj{eta}.Htable

#Set up the table of reciprocal-space radial basis functions.  These
#values are used in the projection-slice theorem so the $\vec k$ of
#interest is a rotation of $(\vec\kappa 0)^T$.  Therefore,
#$|\vec k|=|\vec\kappa|$ and it the the two components of $\vec\kappa$
#that are passed in the argument vk.

def virusobj_set_2Dreciprocal(vobj,vk):
    """ Return: numpy ndarray (matlab cell)
        __author__ = 'Zhengyu Cai'
    """

    #????????? vk is 0   lack some lines about vk statement
    if (vk.size == 0):              # isempty
        h.printf(t.log, 'virusobj_set_2Dreciprocal: vk==[], no change to vobj');
        return;
    if (vk.ndim != 2):              # demension
        h.errPrint('virusobj_set_2Dreciprocal: ndims(vk) %d ~= 2'%(vk.ndim));

    h.printf(t.log, 'virusobj_set_2Dreciprocal: vobj[eta].R1<0.0 means use H_{l,p}(r) on [0,R_2)');
    h.printf(t.log, 'virusobj_set_2Dreciprocal: vobj[eta].R1>=0.0 means use H_{l,p}(r) on (R_1,R_2) with R_1>=0');

    h.printf(t.log, 'virusobj_set_2Dreciprocal: the reciprocal-space radial basis functions are the same for all values of vobj[eta].BasisFunctionType: ');

    for eta in range(len(vobj)):
	h.printf(t.log, "virusobj_set_2Dreciprocal");
	#h.printf(t.log,'virusobj_set_2Dreciprocal: vobj[%d].BasisFunctionType=%d' %(eta,vobj[eta].BasisFunctionType));
    #print vobj[eta].clnp.l

    for eta in range(len(vobj)):
        if (vobj[eta].R1>=0.0):
            h.errPrint('virusobj_set_2Dreciprocal: Only R1<0.0 (not R1>=0.0) is implemented. R1 %g\n' %vobj[eta].R1);

        """
        Below two if statements are to adjust the structure difference
        """
        if len(vobj[eta].clnp.l.shape) != 2:
            vobj[eta].clnp.l = vobj[eta].clnp.l.reshape(-1,1)
        if len(vobj[eta].clnp.p.shape) != 2:
            vobj[eta].clnp.p = vobj[eta].clnp.p.reshape(-1,1)
        temp = np.concatenate((vobj[eta].clnp.l, vobj[eta].clnp.p-1),axis=1)   #vobj data structure not sure
        (vobj[eta].unique_lp, vobj[eta].map_lp2unique, vobj[eta].map_unique2lp) = findUnique(temp);

        #print vobj[eta].map_lp2unique
        #print vobj[eta].unique_lp[:,1]

        vobj[eta].Htable=set_H0table_c(vobj[eta].unique_lp[:,0], vobj[eta].unique_lp[:,1],vk[:,0], vk[:,1], vobj[eta].R2);
        #print "hello"
        #print vobj[eta].Htable

        if (~(np.isreal(vobj[eta].Htable.all()))):
            h.errPrint('virusobj_set_2Dreciprocal: eta %d: Htable is not real!\n',eta);
    return vobj


def set_H0table_c(ll,pp,kappa1,kappa2,rmax):
    """ __author__ = 'Zhengyu Cai' """
    kappa = np.sqrt(format(kappa1)**2+format(kappa2)**2) # modified by Yunhan
    lmax = np.amax(ll)
    pmax = np.amax(pp)

    #ht = np.zeros((ll.shape[0],pp.shape[0]))
    if (ll.shape[0] !=pp.shape[0]):
        h.errPrint('The length of ll %d does not equal to the length of pp %d!' %(ll.shape[0],pp.shape[0]));
    rtmax = 10000.0
    if (rmax > rtmax):
        h.errPrint('rmax %f > rtmax %f in init_hlp0_Hlp0!' %(rmax, rtmax));
    (root,normp) = init_hlp0_Hlp0_c(rmax,lmax,pmax+1,rtmax)
    ht = np.zeros((kappa1.shape[0],ll.shape[0]))
    for jj in range(ll.shape[0]):
        l=ll[jj]
        p=pp[jj] #fprintf(1,'l=%d,p=%d\n',l,p);
        ht[:,jj]=hlpk0_c_vec(l,p+1,kappa,root,rmax).flat
    return ht


def mergeMatrix(clnp_l, clnp_p):
    """"
    this is function is to merge two matrix  like merge([1,2],[3,4])=[1,2,3,4]
    __author__ = 'Zhengyu Cai'
    """
    a = list(clnp_l.transpose())
    b = list(clnp_p.transpose())
    c = [a] + [b]
    c1 = np.array(c)
    return c1.transpose()


def unique_rows(data):
    """"
    this function below is to reduce the duplicate rows in a ndarray
    __author__ = 'Zhengyu Cai'
    """

    data = data.tolist()
    data = np.array(data)   #refresh the ndarray type
    #print data.dtype.descr
    #print data.view(data.dtype.descr * data.shape[1])
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


def findUnique(t):
    """"
    this function is to get the reduced-duplicate ndarray and the relative index of the original rows
    __author__ = 'Zhengyu Cai'
    """
    t1 = unique_rows(t)
    #print t1
    d1 = np.zeros((t1.shape[0],1));
    d1.dtype = np.int32
    d2 = np.zeros((t.shape[0],1));
    d2.dtype = np.int32

    for i in range(t.shape[0]):
        for j in range(t1.shape[0]):
            #if list(t[i,:]) == list(t1[j,:]): # modified by Yunhan
            if np.array_equal(t[i,:], t1[j,:]):
                d2[i,0] = j

    for j in range(t1.shape[0]):
        for i in range(t.shape[0]):
            # if all(list(t[i,:]) == list(t1[j,:])): # modified by Yunhan
            if np.array_equal(t[i,:], t1[j,:]):
                d1[j,0] = i

    return t1,d1,d2

# commented by Yunhan Wang, result not right
# def sphbes(n,x,nargout=4):
#     """ function [sj,sjp,sy,syp]=sphbes(n,x)
#         compute spherical bessel function J-Y, and their derivatives
#         __author__ = 'Zhengyu Cai' """
#     #print np.sqrt(5)
#     if (n<0 or x<0.0):
#         h.errPrint('bad arguments in sphbes : n=%d, x=%f.\n' %(n,x));
#     if (nargout != 4):
#         h.errPrint('bad arguments in sphbes : n=%d, x=%f.\n' %(n,x));
#
#     if x==0:
#         if n==0:
#             sj = 1.0
#         sjp = 0.0
#         sy = 0.0
#         syp = 0.0
#     else:
#         order = n+0.5
#         sj = np.sqrt(np.pi/(2*x)) * s.jv(order,x)
#         sy = np.sqrt(np.pi/(2*x)) * s.yv(order,x)
#
#         sjp = (n/x)*sj-np.sqrt(np.pi/(2*x))*s.jv(order+1,x)
#         syp = (n/x)*sy-np.sqrt(np.pi/(2*x))*s.yv(order+1,x)
#
#     return (sj,sjp,sy,syp)

# commented by Yunhan Wang, result not right
# def jroots_c(l,pmax,xmax):
#     """ translated from jroots.m
#         xmax = rtmax
#         __author__ = 'Zhengyu Cai'  """
#
#     JMAX = 40;
#     k = 1.0e-17
#     #print (float(k))
#     xacc = float(k)
#     p = 0
#     rtn = 0.0
#     pi2 = np.pi/2.0
#     #
#     x = np.zeros((pmax+1, 1));
#     normp = np.zeros((pmax+1, 1));
#     while (p <= pmax and rtn <= xmax):
#         x1 = rtn + pi2;
#         x2 = x1 + pi2;
#         for j in range(JMAX):
#             #print l
#             #print x2
#             (f1,d2,d1,d3) = sphbes(l,x1);
#             (f2,d2,d1,d3) = sphbes(l,x2);
#             if (f1 * f2 <= 0.):
#                 break;
#             x1 = x2;
#             x2 = x1 + pi2;
#         rtn =0.5 * (x1 + x2);
#         for j in range(JMAX):
#             (f,df,sy,syp) = sphbes(l,rtn);
#             dx = f/df;
#             rtn = rtn -dx;
#             if ((x1-rtn) * (rtn-x2) < 0.0):
#                 if (f*f1<0.):
#                     x2 = rtn +dx;
#                     f2 = f;
#                 else:
#                     x1 = rtn + dx;
#                     f1 = f;
#                 rtn = 0.5* (x1 + x2)
#             if (abs(dx) < xacc):
#                 break;
#
#         x[p] = rtn;
#         normp[p] = df / np.sqrt(2.0);
#         p = p + 1;
#     return x,normp

# commented by Yunhan Wang, result not right
# def init_hlp0_Hlp0_c(rmax,lmax,pmax,rtmax):
#     """"
#     initialize variables for hlpr0/Hlpk0
#
#     RuntimeWarning: divide by zero encountered in double_scalars
#       normp[l,p] = 1.0/(normp[l,p] * tmp);
#       __author__ = 'Zhengyu Cai'
#     """
#
#     root = np.zeros((lmax+1, pmax+1));
#     normp = np.zeros((lmax+1, pmax+1));
#     tmp = np.sqrt(rmax) * rmax;
#
#     for l in range(int(lmax+1)):
#         (temp1,temp2) = jroots_c(l,pmax,rtmax)
# #         if l == 0:
# #             print temp1
# #             print temp2
#         for j0 in range(temp1.shape[0]):
#             root[l,j0] = temp1[j0];
#             normp[l,j0] = temp2[j0];
#         for p in range(int(pmax)):
#             normp[l,p] = 1.0/(normp[l,p] * tmp);    # normp =aa
#     return root, normp


def sphbes_vec(n,x,nargout=4):
    """ function [sj,sjp,sy,syp]=sphbes_vec(n,x)
        compute spherical bessel function J-Y, and their derivatives
        __author__ = 'Zhengyu Cai' """

    if (np.isscalar(n)!= True or n < 0):
        h.errPrint('sphbes_vec: n must be scalar >=0')   # for convenience ,  'n=%d.\n' %n' this part missing
    if np.any(x<0.0):
        h.errPrint('sphbes_vec: x must be >=0')   # for convenience, , x=%f.\n' %x

    ix1 = np.where(x != 0.0)[0]
    x1 = x[ix1];
    ix0 = np.where(x == 0.0)[0]


    #Compute sj, sy, sjp, syp for x>0
    fact = np.sqrt(np.pi/2)/np.sqrt(x1)
    order = n + 0.5
    sj1 = np.multiply(fact, s.jv(order, x1))
    sy1 = fact * s.yv(order, x1)


    #by recursion formula
    sjp1 = (n / x1) * sj1 - fact * s.jv(order+1, x1)
    syp1 = (n / x1) * sy1 - fact * s.yv(order+1, x1)


    #Compute sj, sy, sjp, syp for x=0.
    if n == 0:
        sj0 = 1.0
    else:
        sj0 = 0.0
    sjp0 = 0.0
    sy0 = 0.0
    syp0 = 0.0

    Nx = x.shape[0]
    sj = np.zeros((Nx))
    sjp = np.zeros((Nx))
    sy = np.zeros((Nx))
    syp = np.zeros((Nx))

    sj[ix0] = sj0
    sjp[ix0] =sjp0
    sy[ix0] = sy0
    syp[ix0] = syp0
    for j in range(ix1.shape[0]):
        sj[ix1[j]] = sj1[j]
        sjp[ix1[j]] =sjp1[j]
        sy[ix1[j]] =sy1[j]
        syp[ix1[j]] = syp1[j]

    return (sj,sjp,sy,syp)

def hlpk0_c_vec(l, p, k, root, rmax):
    """"
    This function computes the same results as the C version of hlpk0.c but
     the correct version should flip the sign of the "empirical" variable if
     fabs(df) == -df.
    __author__ = 'Zhengyu Cai'
    """
    EPS = 1.0e-6 # modified by Yunhan
    normhlpk0 = np.sqrt(2) * np.sqrt(rmax) * rmax;
    tmp = (2.0 * np.pi *k)*rmax
    empirical = 4.0 * np.pi
    (sj, sjp_junk, sy_junk, syp_junk) = sphbes_vec(l, tmp)

    numer = normhlpk0 * root[l, p-1] *sj;
    denom = tmp *tmp - root[l,p-1]*root[l,p-1];
    numer = numer.reshape(numer.shape[0],1)

    ind1 = np.where(abs(numer) <EPS)[0]
    ind2 = np.where(abs(denom)<EPS)[0]
    ind = np.intersect1d(ind1,ind2)

    if (ind.size != 0):
        # use L'Hospital's Rule
        for j in range(ind.shape[0]):

            numer[ind[j]] = normhlpk0 * root[l,p-1] *sjp_junk[ind[j]];
            denom[ind[j]] = 2.0 * tmp[ind[j]]
        h.errPrint('hlpk0_c_vec: bad, l %d p %d k %g\n' %(l,p,k))
        # this line below is not sure about the correctness
        h.errPrint('hlpk0_c_vec: bad, EPS %f normhlpk0 %f root(l+1,p) %f sjp_junk %f tmp %f\n' %(EPS,normhlpk0,root[l,p-1],sjp_junk,tmp))
        h.errPrint('should never reach this statement!\n')
    y = empirical*numer / denom # modified by Yunhan
    return y;

############ 2b part #############
def load(fp, *strings):
    """ load arrays whose names are in strings from .mat
    __author__ = 'Yunhan Wang' """
    f = io.loadmat(fp, squeeze_me=True, struct_as_record=False)
    lists = []
    for st in strings:
        lists.append(f[st])
    return lists

def box_normalize_mask(dimensions,delta,radius):
    """ dimensions(1:2) -- non-negative integers, dimensions of the image
    delta(1:2) -- positive reals, sampling intervals
    radius(1:2) -- positive reals, radius(1) is the inner radius and
    radius(2) is the outter radius of the annulus
    __author__ = 'Yunhan Wang' """
    # dimensions = [91,91]
    tmp1 = np.array(range(dimensions[0]))-(dimensions[0]-1)/2
    tmp2 = np.array(range(dimensions[1]))-(dimensions[1]-1)/2
    X2,X1 = np.meshgrid(delta[0] * tmp1, delta[1] * tmp2) # TODO test
    R = (X1**2 + X2**2)**0.5
    mask = np.logical_and(radius[0]<=R, R<radius[1]).astype(bool)
    return mask

def virusobj_changesize(vlmax,vpmax,old_vobj):
    Neta = vlmax.size
    new_vobj = old_vobj
    for eta in range(Neta):
        lmax = vlmax[eta]
        pmax = vpmax[eta]
        llist = np.zeros(((lmax+1)**2,1))
        nlist = np.zeros(((lmax+1)**2,1))
        #the following code causes n to change more rapidly than l
        ln=-1
        for l in range(lmax+1):
            for n in range(2*l+1):
                if h.slct(l,n)==1: #selection rule for icosahedrally-symmetric harmonic angular basis functions
                    ln=ln+1
                    llist[ln][0]=l
                    nlist[ln][0]=n
        h.printf(t.log,'virusobj_changesize: eta %d ln %d'%(eta+1,ln+1))
        new_il=np.zeros(((ln+1)*pmax,1),float)
        new_in=np.zeros(((ln+1)*pmax,1),float)
        new_ip=np.zeros(((ln+1)*pmax,1),float)
        new_optflag=np.ones(((ln+1)*pmax,1),float) #default for optflag is 1
        new_c=np.zeros(((ln+1)*pmax,1),float) #default for clnp is 0.0
        if not (old_vobj[eta].nu.size == 0):
            new_nu=np.zeros(((ln+1)*pmax,1),float) #default for nu is 0.0
        else:
            new_nu=np.array([])
        lnp=-1
        for ii in range(ln+1):
            for p in range(pmax):
                lnp=lnp+1
                new_il[lnp][0]=llist[ii][0]
                new_in[lnp][0]=nlist[ii][0]
                new_ip[lnp][0]=p+1
        old_il=old_vobj[eta].clnp.l
        old_in=old_vobj[eta].clnp.n
        old_ip=old_vobj[eta].clnp.p
        old_optflag=old_vobj[eta].clnp.optflag
        old_c=old_vobj[eta].clnp.c
        old_nu=old_vobj[eta].nu

        for ii in range(lnp+1):
            # below are all 1*1 matrix
            logical_l=(old_il==new_il[ii][0])
            logical_n=(old_in==new_in[ii][0])
            logical_p=(old_ip==new_ip[ii][0])
            logical_lnp=(logical_l & logical_n & logical_p)
            jj =  (logical_lnp==True).nonzero()
            if jj[0].size>1:
                print 'virusobj_changesize: ii %d found duplicates\n'%ii

            if jj[0].size == 1:
                new_optflag[ii]=old_optflag[jj[0][0]]
                new_c[ii][0]=old_c[jj[0][0]]
                if not (old_vobj[eta].nu.size == 0):
                    new_nu[ii]=old_nu[jj[0][0]]
        #replace existing values
        new_vobj[eta].clnp.l = new_il
        new_vobj[eta].clnp.n = new_in
        new_vobj[eta].clnp.p = new_ip
        new_vobj[eta].clnp.optflag = new_optflag
        new_vobj[eta].clnp.c = new_c
        new_vobj[eta].nu = new_nu
        new_vobj[eta].cbar = new_vobj[eta].clnp.c #for convenience

        new_vobj[eta].clnp_fn = new_vobj[eta].clnp_fn + '+'
        new_vobj[eta].nu_fn = new_vobj[eta].nu_fn + '+'
        new_vobj[eta].q_fn = new_vobj[eta].q_fn + '+'
    return new_vobj

def virusobj_change_homo2hetero(vobj,homo2hetero):
    """ __author__ = 'Xiao Ma' """
    for eta in range(len(vobj)):
        if homo2hetero[eta].action==1: #make this class heterogeneous
            h.printf(t.log,'virusobj_change_homo2hetero: eta %d: make heterogeneous'%(eta))
            vobj[eta].nu = mean2stddev(vobj[eta].cbar,homo2hetero[eta].FractionOfMeanForMinimum,homo2hetero[eta].FractionOfMean)**2

        elif homo2hetero[eta].action==0: #no change
            h.printf(t.log,'virusobj_change_homo2hetero: eta %d: no change'%(eta))

        elif homo2hetero[eta].action==-1: #make this class homogeneous
            h.printf(t.log,'virusobj_change_homo2hetero: eta %d: make homogeneous'%(eta))
            vobj[eta].nu = []

        else:
            h.errPrint('virusobj_change_homo2hetero: eta %d unknown operator %s'%(eta,homo2hetero[eta].action))
    return vobj

def mean2stddev(cbar,FractionOfMeanForMinimum,FractionOfMean):
    """ __author__ = 'Xiao Ma' """
    if FractionOfMeanForMinimum<0.0:
        h.errPrint('mean2stddev: FractionOfMeanForMinimum %g < 0.0'%(FractionOfMeanForMinimum));
    if FractionOfMean<0.0:
        h.errPrint('mean2stddev: FractionOfMean %g < 0.0'%(FractionOfMean));

    ''''
    Zhye Yin JSB 2003 paper: compute minimum from l=0,n=0,p=1 weight, assume that l=0,n=0,p=1 weight is cbar(1).
    minstddev=FractionOfMeanForMinimum*abs(cbar(1));
    Could search for the l=0,n=0,p=1 element -- have not written code.
    Here I use the largest element of cbar.
    '''
    minstddev=FractionOfMeanForMinimum*max(abs(cbar));

    stddev=FractionOfMean*np.absolute(cbar);
    last=np.amax(stddev);
    for nNc in range(len(stddev)):
        if stddev[nNc]==0.0:
            stddev[nNc]=last
        if stddev[nNc]<minstddev:
            stddev[nNc]=minstddev
        last=stddev[nNc]
    return stddev

def subset_reciprocalspace(kmax,vkmag,vkminimalset,Imagestack,iiminimalset):
    # kmax = -1, 4141*1, 4141*2, 50*1, 4141*1
    """ Construct the reciprocal space image data structure for the range of reciprocal space
    $\|\vec\kappa\|<kmax$ that will be used in this step.
    I believe we continue to separately use both $\Re Y(\vec\kappa=0)$ and $\Im Y(\vec\kappa=0)$
    even though $\Im Y(\vec\kappa=0)=0$.
    __author__ = 'Yunhan Wang' """
    if kmax == -1: # use everything
        vkindex = np.asarray(range(len(vkmag)))
        vk = vkminimalset
    elif kmax==-2: #use everything except |\vec\kappa|=0
        vkindex = find_operators(vkmag, 0, '>')
        vk = vkminimalset[vkindex, :]
    elif 0 <= kmax: # general case
        vkindex=find_operators(vkmag, kmax, '<')
        vk=vkminimalset[vkindex,:]
    else: # error case
        h.errPrint('subset_reciprocalspace: kmax %g'%(kmax))
    Ny = vk.shape[0]
    # Variable y will store each image in consecutive memory; matlab uses column-major storage
    y=np.zeros((Ny*2,len(Imagestack)),float)
    for ii in range(len(Imagestack)):
        # change to 1 dimension for indexing: row major->column major
        Isflat = Imagestack[ii][0].T.flat
        y[0::2,ii] = Isflat[iiminimalset[vkindex]].real.flat
        y[1::2,ii] = Isflat[iiminimalset[vkindex]].imag.flat
    return [vk,y]

def setL_nord(Rabc, ll, mm, vk, Htable, map_unique2lp, tilde_b):
    """ __author__ = 'Zhengyu Cai' """
    Nc = ll.shape[0]
    Ny = vk.shape[0] # modified by Yunhan
    L = np.zeros((Ny*2, Nc)) # modified by Yunhan
    kk = np.append(vk, np.zeros((Ny, 1)), axis=1)
    kknew = np.dot(kk,Rabc.conj())

    sphkk = xyztosph_vec(kknew)

    Atable=set_Ttable_nord(ll,mm,sphkk[:,1],sphkk[:,2],tilde_b)

    if len(map_unique2lp.shape) < 2:
        d1 = map_unique2lp.shape[0]
        map_unique2lp = scipy.reshape(map_unique2lp, [d1, 1])
    # set up L
    # index offset in L and Htable modified by Guantian
    for ii in xrange(0,Nc):
        mod_num = ll[ii] % 4
    	if mod_num == 0: # (-i)^0=1
            """
            ATTENTION:
            map_unique2lp in matlab is a N*1 vector, however, in python it's a N*2, the second column is zero
            """
            L[0:Ny*2-1:2,ii] = Atable[ii].flatten(1)*Htable[:,map_unique2lp[ii][0]] # real part for Ttable
            """
            L(1:2:Ny*2-1,ii) = Atable{ii}(mm(ii)+1,:).*Htable(:,map_unique2lp(ii))'; % real part for Ytable
            """
        elif mod_num == 1: # (-i)^1=-i
            L[1:Ny*2:2,ii] = -Atable[ii].flatten(1)*Htable[:,map_unique2lp[ii][0]] # imaginary part
            """
            L(2:2:Ny*2,ii) = -Atable{ii}(mm(ii)+1,:).*Htable(:,map_unique2lp(ii))';
            """
        elif mod_num == 2: # (-i)^2=-1
            L[0:Ny*2-1:2,ii] = -Atable[ii].flatten(1)*Htable[:,map_unique2lp[ii][0]] # real part
            """
            L(1:2:Ny*2-1,ii) = -Atable{ii}(mm(ii)+1,:).*Htable(:,map_unique2lp(ii))';
            """
        elif mod_num == 3: # (-i)^3=i
            L[1:Ny*2:2,ii] = Atable[ii].flatten(1)*Htable[:,map_unique2lp[ii][0]] # imaginary part
            """
            L(2:2:Ny*2,ii) = Atable{ii}(mm(ii)+1,:).*Htable(:,map_unique2lp(ii))';
            """
        else:
            sys.exit('Incorrect value of mod(ll(ii),4) %d!\n'%mod_num)
    return L

def assert_UpdateBothCopiesOfcbar(vobj):
    """ __author__ = 'Yunhan Wang' """
    for eta in range(len(vobj)):
        #if any( (vobj[eta].cbar-vobj[eta].clnp.c)~=0.0 ):
            h.errPrint('assert_UpdateBothCopiesOfcbar: eta %d vobj{eta}.cbar ~= vobj{eta}.clnp.c'%(eta))

def EM_expmax_MonteCarlo(vk,y,vobj,pixelnoisevar,EM_MC,EM_iter,all_tilde_b):
    """ __author__ = 'Yunhan Wang' """
    if EM_MC.Nrandic <= 0:
        h.errPrint('EM_expmax_MonteCarlo: EM_MC.Nrandic %d < 0'%(EM_MC.Nrandic))
    h.printf(t.log,'EM_expmax_MonteCarlo: Nrandic %d'%(EM_MC.Nrandic))
    (vobj_best,pixelnoisevar_best,loglikelihood_best,toomanyiterations)=EM_expmax_iterate2convergence(vk,y,vobj,pixelnoisevar,EM_iter,all_tilde_b)

    assert_UpdateBothCopiesOfcbar(vobj)
    if EM_MC.Nrandic==1:
        return vobj_best,pixelnoisevar_best,loglikelihood_best

    Neta=len(vobj)
    # EM_expmax_iterate2convergence computes the log likelihood only for homogeneous problems.
    # So Monte Carlo applied to the initial conditions is only permitted for homogeneous problems.
    for eta in range(Neta):
        if len(vobj[eta].nu) != 0:
            h.errPrint('EM_expmax_MonteCarlo: eta %d The case of multiple initial conditions for heterogeneous problems is not implemented'%(eta+1))

    # Nc(eta) can vary with eta so use a cell array, not a 2-D matrix
    cbar_ic_mean = np.zeros((Neta,1), dtype=np.object)
    cbar_ic_stddev = np.empty((Neta,1), dtype=np.object)
    for eta in range(Neta):
        cbar_ic_mean[eta][0]=vobj[eta].cbar
        cbar_ic_stddev[eta][0] = mean2stddev(cbar_ic_mean[eta][0],EM_MC.FractionOfMeanForMinimum,EM_MC.FractionOfMean)
        h.printf(t.log,'EM_expmax_MonteCarlo: eta %d cbar_ic_mean(eta):'%(eta+1))
        print cbar_ic_mean[eta][0].T
        h.printf(t.log,'EM_expmax_MonteCarlo: eta %d cbar_ic_stddev(eta):'%(eta+1))
        print cbar_ic_stddev[eta][0].T

    for nMC in range(1,EM_MC.Nrandic):
        h.printf(t.log,'EM_expmax_MonteCarlo: nMC %d'%(nMC+1))
        for eta in range(Neta):
            shape_tuple = cbar_ic_stddev[eta][0].shape
            vobj[eta].cbar = cbar_ic_mean[eta][0] + cbar_ic_stddev[eta][0]*np.random.randn(shape_tuple[0],shape_tuple[1])
            vobj[eta].clnpc = vobj[eta].cbar
    (vobj,pixelnoisevar,loglikelihood,toomanyiterations)=EM_expmax_iterate2convergence(vk,y,vobj,pixelnoisevar,EM_iter,all_tilde_b)
    if toomanyiterations != None:
        pass
    assert_UpdateBothCopiesOfcbar(vobj)

    if loglikelihood > loglikelihood_best:
        h.printf(t.log,'EM_expmax_MonteCarlo: nMC %d current answer is best so far loglikelihood %g loglikelihood_best %g'%(nMC+1,loglikelihood,loglikelihood_best))
        loglikelihood_best=loglikelihood
        vobj_best=vobj
        pixelnoisevar_best=pixelnoisevar
    else:
        h.printf(t.log,'EM_expmax_MonteCarlo: nMC %d current answer is not best so far loglikelihood %g loglikelihood_best %g'%(nMC+1,loglikelihood,loglikelihood_best))
    # end of nMC=2:EM_MC.Nrandic
    return vobj,pixelnoisevar,loglikelihood

######### 3b part ########
################ post-processing operators: ################
def plt_realspace(wantrhobar,wantrxx,onevobj,onetilde_b,m1l,m1h,d1,m2l,m2h,d2,m3l,m3h,d3,alpha,beta,gamma):
    """ Precondition: onevobj is read from .mat, that is, the field in vobj is .il, .in, rather than .l, .n in SingleVirus()
    __author__ = 'Yunhan Wang' """
    # This function is almost able to also compute 3-D reciprocal space
    # cubes because DoerschukJohnson IEEE T-IT 2000 Eq. 3 is so similar to
    # Eq. 2.  The angular basis functions are the same, but I might change
    # the name of the variable 'angular' to 'aAngular' to emphasize this
    # point.  The radial basis functions must be changed: replace
    # hlpr0_c_vec by hlpk0_c_vec and then multiply each basis function or
    # each c_{l,n,p} value by (-1)^l.  This multiplication is not difficult
    # because the basis functions and c_{l,n,p} are both real.  However,
    # the answer will be complex so rhobar would be complex.  That is not a
    # problem in matlab, but maybe I would do something else because of an
    # interest in C.  For rxx there is more to think about because there
    # are really multiple correlation functions depending on where you put
    # \Re and \Im or ^\ast.  I have done similar calculations in the past,
    # e.g., LeeDoerschukJohnson IEEE T-IP 2007 Eqs. 51-54.

    # Set in case they are not computed.
    rhobar,rxx,x_rect=np.ndarray(shape=(0,0), dtype=float),np.ndarray(shape=(0,0), dtype=float), np.ndarray(shape=(0,0), dtype=float)
    R2 = onevobj.R2
    ll = onevobj.clnp.l
    mm = onevobj.clnp.n #Spherical harmonics notation of 'm' not icosahedral harmonics notation of 'n'
    pp = onevobj.clnp.p
    c = onevobj.clnp.c
    nu = onevobj.nu
    WhichAngularBasis = onevobj.BasisFunctionType
    tilde_b = onetilde_b.tilde_b

    IcosahedralBasis, SphericalBasis = 1,2

    # Initialize variables for hlpr0/hlpk0
    rtmax = 10000.0
    lmax, pmax = np.amax(ll), np.amax(pp)
    if R2 > rtmax:
        h.errPrint('plt_realspace: R2 %g > rtmax %g'%(R2,rtmax))
    root,aa = init_hlp0_Hlp0_c(R2,lmax,pmax,rtmax) # TODO: slow!! b/c jroots_c
    # root = load('root.mat', 'root')[0]
    # aa = load('aa.mat', 'aa')[0]

    # Setup the grid
    mlh1 = [i for i in range(m1l, m1h+1)]
    mlh2 = [i for i in range(m2l, m2h+1)]
    mlh3 = [i for i in range(m3l, m3h+1)]
    m1,m2,m3 = map(inverse3d, np.meshgrid(mlh1,mlh2,mlh3))
    m123 = np.asarray([m1.flatten()*d1, m2.flatten()*d2, m3.flatten()*d3]).T
    Rinv = euler2R([-gamma,-beta,-alpha])
    x_rect = np.dot(m123, Rinv) # rotate the coordinates by Rinv
    x_sph = xyztosph_vec(x_rect) # convert rectangular coordinates to spherical coordinates
    unique_lp, map_lp2unique, map_unique2lp = unique2(zip(ll,pp), "row")
    unique_radius, map_radius2unique, map_unique2radius = unique2(x_sph[:,0], "")
    htable=np.zeros((len(unique_radius),len(unique_lp)), float)
    for lp in range(len(unique_lp)):
        htable[:,lp]=hlpr0_c_vec(unique_lp[lp,0],unique_lp[lp,1],unique_radius,root,aa,R2).flat
    # rhobar: ZhengWangDoerschuk JOSA-A 2012 Eq. 16.
    # rxx: ZhengWangDoerschuk JOSA-A 2012 Eq. 18 with $\vec x^\prime=\vec x$ and
    # $(V_{\eta^\prime})_{\tau,\tau^\prime}=\nu_{\eta^\prime}(\tau)\delta_{\tau,\tau^\prime}$.
    if wantrhobar:
        rhobar=np.zeros((x_sph.shape[0],1), float)
    if wantrxx:
        rxx= np.zeros((x_sph.shape[0],1), float)
    for ii in range(len(ll)):
        l=ll[ii]
        m=mm[ii]
        # Not yet implemented: reuse h when l and p have not changed from the previous iteration.
        h=htable[:,map_unique2lp[ii]]
        h_full=h[map_unique2radius]

        #Reuse angular if ii>1 && l==ll(ii-1)
        if ii==0 or l!=ll[ii-1]:
            if WhichAngularBasis==IcosahedralBasis:
                angular=set_Ttable_nord(l,m,x_sph[:,1],x_sph[:,2],tilde_b)
            elif WhichAngularBasis==SphericalBasis:
		print
            else:
                h.errPrint('plt_realspace: WhichAngularBasis %d #1'%(WhichAngularBasis))
        if WhichAngularBasis==IcosahedralBasis:
            if wantrhobar:
                rhobar=rhobar + c[ii]*( (h_full[:, np.newaxis]*(angular[0][:]))    )
            if wantrxx:
                rxx=rxx + nu[ii]*(      (h_full[:, np.newaxis]*(angular[0][:]))**2 )
        elif WhichAngularBasis==SphericalBasis:
            if wantrhobar:
                rhobar=rhobar + c[ii]*( (h_full[:, np.newaxis]*(angular[0][mm[ii],:].T[:, np.newaxis]))    )
            if wantrxx:
                rxx=rxx + nu[ii]*(      (h_full[:, np.newaxis]*(angular[0][mm[ii],:].T[:, np.newaxis]))**2 )
        else:
            h.errPrint('plt_realspace: WhichAngularBasis %d #2'%(WhichAngularBasis))
    # close for ii=1:length(ll)

    if rhobar.size != 0:
        rhobar=np.reshape(rhobar,(m1h-m1l+1,m2h-m2l+1,m3h-m3l+1))
    if rxx.size != 0:
        rxx=np.reshape(rxx,(m1h-m1l+1,m2h-m2l+1,m3h-m3l+1))
    return rhobar,rxx,x_rect

def init_hlp0_Hlp0_c(rmax,lmax,pmax,rtmax):
    """ initialize variables for hlpr0/Hlpk0
    __author__ = 'Yunhan Wang' """
    rmax = int(rmax)
    lmax = int(lmax)
    pmax = int(pmax)
    rtmax = int(rtmax)
    # initialize
    root=np.zeros((lmax+1,pmax), float)
    normp=np.zeros((lmax+1,pmax), float)
    tmp = np.sqrt(rmax) * rmax
    for l in range(lmax+1):
        root[l,:], normp[l,:]=jroots_c(l,pmax,rtmax) # TODO: very slow
        for p in range(pmax):
            normp[l,p] = 1.0 / (normp[l,p] * tmp) # normp=aa
    return root,normp

def jroots_c(l,pmax,xmax):
    """ xmax = rtmax
     Translated from jroots.c
     Y. Zheng
     2/13/2008
     TODO: these python codes are very low effecient, should be improved
     __author__ = 'Yunhan Wang' """

    JMAX=40
    xacc=1.0e-17
    p = 0
    rtn = 0.0
    pi2 = pi/2.0
    x = np.zeros((pmax,1), float)
    normp = np.zeros((pmax,1), float)
    while (p < pmax and rtn <= xmax):
        x1 = rtn + pi2
        x2 = x1 + pi2
        for j in range(JMAX):
            f1,d2,d1,d3 = sphbes(l,x1)  #sphbes(l, x1, &f1, &d1, &d2, &d3)
            f2,d2,d1,d3 = sphbes(l,x2)  #sphbes(l, x2, &f2, &d1, &d2, &d3);
            if (f1 * f2 <= 0):
                break
            x1 = x2
            x2 = x1 + pi2
        rtn = 0.5 * (x1 + x2)
        for j in range(JMAX):
            f,df,sy,syp = sphbes(l,rtn) #sphbes(l, rtn, &f, &sy, &df, &syp);
            dx = f / df
            rtn = rtn - dx
            if ((x1 - rtn) * (rtn - x2) < 0.0):
                if(f * f1 < 0.):
                    x2 = rtn + dx
                    f2 = f
                else:
                    x1 = rtn + dx
                    f1 = f
                rtn = 0.5 * (x1 + x2)
            if (abs(dx) < xacc):
                break
        x[p] = rtn
        normp[p] = df / sqrt(2.0)   # /* 10 March 07: sqrt(z*z)=fabs(z) not z */
        # The following fix doesn't work with the clnp's from the current hlpk0_c normp(p) = abs(df) / sqrt(2.0)
        # /* 10 March 07: sqrt(z*z)=fabs(z) not z */
        p = p + 1
    return x.flatten(), normp.flatten()

def sphbes(n,x):
    """ function [sj,sjp,sy,syp]=sphbes(n,x)
    compute spherical bessel function J-Y, and their derivatives
    modified from /home/rupee/a/doerschu/radial_basis_funcs2/sphebes.real.c
    incomplete treatment for x=0
    __author__ = 'Yunhan Wang' """
    nargout = 4 # TODO: number of output arguments that were used to call the function?
    if n<0 or x<0.0:
        h.errPrint('bad arguments in sphbes : n=%d, x=%g.'%(n,x))
    if nargout !=4:
        h.errPrint('number of outputs should be two or four.')

    if x==0.0:
        if n==0:
            sj=1.0
        else:
            sj=0.0
        sjp=0.0
        sy=0.0
        syp=0.0
    else:
        order=n+0.5
        sj=sqrt(pi/(2*x))*besselj(order,x)
        sy=sqrt(pi/(2*x))*bessely(order,x)
        # by recursion formula
        sjp=(n/x)*sj-sqrt(pi/(2*x))*besselj(order+1,x)
        syp=(n/x)*sy-sqrt(pi/(2*x))*bessely(order+1,x)
    return sj,sjp,sy,syp

def inverse3d(array):
    """ rotate the 3D arr like this:
      [::1] = 0,1;0,2;0,3
      [::2] = 1,4;1,5;1,6
      output
      [::1] = 0,0,0;1,1,1
      [::2] = 1,2,3;4,5,6
    __author__ = 'Yunhan Wang' """
    arr = array.T
    T_inner = [] # inner transpose of arr
    for cell in arr:
        T_inner.append(cell)
    return np.asarray(T_inner)

def xyztosph_vec(x):
    """ convert cartesian coordinates to spherical coordinates
    ectorized version of xyztosph.m, which is modified from xyztosph.c
    __author__ = 'Yunhan Wang' """
    if x.shape[1] != 3:
        h.errPrint('x should be a n-by-3 array.')
    r,theta,phi = np.zeros((x.shape[0]), float),np.zeros((x.shape[0]), float),np.zeros((x.shape[0]), float)
    r[:] = np.sqrt(x[:,0]*x[:,0]+x[:,1]*x[:,1]+x[:,2]*x[:,2])
    for i in range(x.shape[0]):
        if r[i]==0.0:
            theta[i] = phi[i] = 0.0
        else:
            theta[i] = math.acos(x[i,2]/r[i])
            if x[i,0]==0.0 and x[i,1]==0.0:
                phi[i] = 0.0
            else:
                phi[i] = math.atan2(x[i,1], x[i,0])
    return np.asarray([r,theta,phi]).T

def euler2R(t):
    """ convert rectangular coordinates to spherical coordinates
    t=(alpha,beta,gamma)
    __author__ = 'Yunhan Wang' """
    ca = cos(t[0])
    sa = sin(t[0])
    cb = cos(t[1])
    sb = sin(t[1])
    cg = cos(t[2])
    sg = sin(t[2])

    R = np.zeros((3,3), float)

    R[0,0] = ca * cb * cg - sa * sg
    R[0,1] = sa * cb * cg + ca * sg;
    R[0,2] = -sb * cg;

    R[1,0] = -ca * cb * sg - sa * cg;
    R[1,1] = -sa * cb * sg + ca * cg;
    R[1,2] = sb * sg;

    R[2,0] = ca * sb;
    R[2,1] = sa * sb;
    R[2,2] = cb;
    return R

def hlpr0_c_vec(l,p,rr,root,aa,R2):
    """ This is the vectorized version of hlpr0_c.m
    hlpr0_c.m compute radial basis function hlp(r) when inner radius R1 is zero
    modified from /home/rupee/a/doerschu/radial_basis_funcs2/Hlpr0.c
    Yili Zheng
    05/19/2008

    Caustion: hlpr0_c_vec.m and hlpr0_c.m need to be used in
    conjunction with jroots_c.m and other *_c.m which are translated
    from their C counterparts.

    variable naming convention: R2=Rmax, r=rr, aa=normp

    Please note that the _c version  differs from Seunghee's version in two

    1) Seunghee's roots = _c version's roots/Rmax;
    hlp0_c divides root*rr by R2:
      [sj,sjp,sy,syp]=sphbes_vec(l,root(l+1,p).*rr(rr_nz_ind)/R2);
    but Seunghee's hlpr0.m doens't divide root*r by R2 because root has
    already been divided by R2 in jroots.m:
      [sj,sjp,sy,syp]=sphbes(l,root(l+1,p)*r);
    .
    This difference would not change the final answer.

    2) aa in Seunghee's code is always positive but not in the _c version.

    The C version code works because the signs of the coefficients are the
    opposite in both hlpk0_c.m and hlpr0_c.m for the "l,p" pairs that have
    negative "df".
    __author__ = 'Yunhan Wang' """

    rr_nz_ind = np.logical_and(rr>=0,rr<=R2)
    rr_zero_ind = np.logical_not(rr_nz_ind) #If redundant below, then this is also redundant.
    sj,sjp,sy,syp = sphbes_vec(l,root[l,p-1]*rr[rr_nz_ind]/R2)
    y2=np.dot(aa[l,p-1], sj)

    Nr = rr.shape[0]
    y = np.zeros((Nr,1),float)
    y[rr_zero_ind]=0.0 # Redundant with the initialization by 'zeros'?
    y[rr_nz_ind]=y2[:, np.newaxis]
    return y

def unique3(a, method):
    """ __author__ = 'Yunhan Wang' """
    if method == "row":
        a1 = a.tolist()
        c = np.unique(a.view([('',a.dtype)]*a.shape[1])).view(a.dtype).reshape(-1,a.shape[1])
    else:
        a1 = a
        c = np.unique(a)
    c1 = c.tolist()
    ia, ic = [], []
    l = len(a1)
    for i in c1:
        ia.append(l-a1[::-1].index(i)-1)
    for i in a1:
        ic.append(c1.index(i))
    return c, ia, ic

def unique2(a1, method):
    """ __author__ = 'Yunhan Wang' """
    ia, ic = [], []
    if method == "row":
        a = a1
        c = np.unique(a)
        c1 = c.tolist()

        for i in c1:
            ia.append(a.index(tuple(i)))
        for i in a:
            ic.append(c1.index(list(i)))
        return c, ia, ic

    else:
        a = a1.T.flatten().tolist()
        c = np.unique(a)
    c1 = c.tolist()

    for i in c1:
        ia.append(a.index(i))
    for i in a:
        ic.append(c1.index(i))
    return c, ia, ic

def set_Ttable_nord(ll, nn, theta, phi,tilde_b):
    """ Compute icosahedral harmonics basis functions
    See double ti(int l, int n, double phi, double theta) in ti.c

    Yili Zheng
    06/25/2008

    b_fn = 'callti.out.read_by_C'; # file contains \tilde{b} values computed from mathematica
    lmax = max(ll);
    tilde_b = rd_b(b_fn, lmax);
    __author__ = 'Yunhan Wang' """

    if type(ll) is np.uint8:
        ll = np.asarray([ll])
    if type(nn) is np.uint8:
        nn = np.asarray([nn])
    Nc = len(ll)
    Ny = len(theta)
    assert (len(phi)==Ny)
    Ttable = []
    costheta=np.cos(theta)
#     print "nn is %d %d"%(nn.shape[0],nn.shape[1])
    prev_l = -1
    prev_n = -1
    for ii in range(Nc):
        l = int(ll[ii])
        n = int(nn[ii]) # nn[ii][0], conditional assignment, by Yunhan Wang 08/14/2014
        if l==prev_l and n==prev_n:
            Ttable.append(Ttable[ii-1])
        else:
            # plgndrtable = legendre(l,costheta);
            # Tlm is stored in the m+1 index because the first index in Matlab is 1
            # but m starts with 0.
            Ttable.append(np.zeros((Ny,1),float))

            if (l%2==0): # l even
                #for m in range(n,l/5+1):
                #print(l)
                for m in range(n,l/5+1):

                    mprime = m*5
                    Nlm = np.sqrt((2*l+1)*math.factorial(l-mprime)/(4*math.pi*math.factorial(l+mprime)))
                    # IF statement added by Guantian
                    # tilde_b might be with demension less than 3

                    if len(tilde_b.shape) < 3:
                        d1, d2 = tilde_b.shape
                        tilde_b = scipy.reshape(tilde_b, [d1, 1, d2])

                    if m==0:
                        # Ttable{ii}(:) = Ttable{ii}(:) + Nlm.*tilde_b(l+1,n+1,m+1).*plgndrtable(mprime+1,:)'.*cos(mprime*phi);
                        # print "l,n,m:%d %d %d"%(l,n,m)
                        #print(ii)
                        #print("ii"+str(ii)+"l"+str(l))
                        Ttable[ii][:] = Ttable[ii][:] + (Nlm*tilde_b[l,n,m]*(plgndr(l,mprime,costheta))*np.cos(mprime*phi)[:,np.newaxis])

                        # Ttable{ii}(:,n) = Ttable{ii}(:,n) + Nlm.*tilde_b(l+1,n+1,m+1).*plgndr(l,mprime,costheta).*cos(mprime*phi);
                    else:
                        #Ttable{ii}(:) = Ttable{ii}(:) + 2*Nlm.*tilde_b(l+1,n+1,m+1).*plgndrtable(mprime+1,:)'.*cos(mprime*phi);

                        Ttable[ii][:] = Ttable[ii][:] + (2*Nlm*tilde_b[l,n,m]*plgndr(l,mprime,costheta)*np.cos(mprime*phi)[:,np.newaxis])
            else: # l odd
                for m in range(n+1, l/5+1):
                    mprime = m*5
                    Nlm = np.sqrt((2*l+1)*math.factorial(l-mprime)/(4*math.pi*math.factorial(l+mprime)))
                    #Ttable{ii}(:) = Ttable{ii}(:) + 2*Nlm*tilde_b(l+1,n+1,m+1)*plgndrtable(mprime+1,:)'.*sin(mprime*phi);
                    Ttable[ii][:] = Ttable[ii][:] + (2*Nlm*tilde_b[l,n,m]*(plgndr(l,mprime,costheta))*(np.sin(mprime*phi)[:,np.newaxis]))
            prev_l = l
            prev_n = n
    return Ttable

def plgndr(l, m, x):
    """ Translated from plgndr.c
    See Numerical Recipes Ch. 6.7

    x can be a scalar or a vector

    Yili Zheng
    6/25/2008
    __author__ = 'Yunhan Wang' """
    if (m < 0 or m > l or np.count_nonzero((np.abs(x)>1.0).astype(int))):
        h.errPrint('Bad arguments in routine plgndr')

    size = x.shape
    if len(size) == 1:
        size = (size[0],1)
    pmm = np.ones(size)

    if m > 0:
        somx2 = np.sqrt((1.0 - x)*(1.0 + x))
        fact = 1.0;
        for i in range(1,m+1):
            pmm = pmm*(-fact * somx2[:,np.newaxis])
            fact = fact + 2.0
    if l == m:
        plm = pmm
        return plm
    else:
        pmmp1 = x[:,np.newaxis]*(2 * m + 1)*pmm
        if l == (m + 1):
            plm = pmmp1
            return plm
        else:
            for ll in range(m + 2,l+1):
                pll = (x[:,np.newaxis]*(2 * ll - 1)*(pmmp1) - (ll + m - 1) * pmm) / (ll - m)
                pmm = pmmp1
                pmmp1 = pll
            plm = pll
    return plm

def set_Ytable(ll,theta,phi):
    """ __author__ = 'Yunhan Wang' """
    # needs further tests, could change all shape of (N,) to (N,1) using format()
    if type(ll) is np.uint8:
        ll = np.asarray([ll])
    Nc=len(ll)
    Ny=len(theta)
    assert (len(phi)==Ny)
    Ytable = []
    costheta=np.cos(theta)
    sqrt2 = sqrt(2)

    for ii in range(Nc):
        l = ll[ii]
        if ii>0 and l==ll[ii-1]:
            Ytable.append(Ytable[ii-1])
        else: # TODO: legendre not working!!! these codes are not used in the test case
            plgndrtable = legendre(l, costheta[:,np.newaxis])
            # Ylm is stored in the m+1 index because the first index in Matlab is 1
            # but m starts with 0.
            Ytable.append(np.zeros((2*l+1,Ny), float))
            # m=0 case
            Nl0 = np.sqrt((2*l+1)/(4*math.pi))
            Ytable[ii][0,:] = Nl0*plgndrtable[0,:]
            for m in range(2, 2*l+1, 2):
                Nlm = np.sqrt((2*l+1)*math.factorial(l-m/2)/(4*math.pi*math.factorial(l+m/2)))
                m_2_phi = m/2*phi[:]
                Ytable[ii][m-1,:] = (sqrt2*Nlm*plgndrtable[m/2,:].T*np.sin(m_2_phi)[:,np.newaxis]).flat #m={1,3,...,2l-1}
                Ytable[ii][m,:] = (sqrt2*Nlm*plgndrtable[m/2,:].T*np.cos(m_2_phi)[:,np.newaxis]).flat #m={2,4,...,2l}
    return Ytable

def size(a):
    """ __author__ = 'Yunhan Wang' """
    s = a.shape
    if len(s) == 1: # (N,)
        return [s[0],1]
    return s # (N1, N2)

def format(a):
    """ format an array: transfer its shape from (N,) to (N,1)
    __author__ = 'Yunhan Wang' """
    if len(a.shape) == 1:
        return a[:, np.newaxis]
    return a

def get_FSC(onevobjA,onevobjB,minmagk,maxmagk,deltamagk):
    """ __author__ = 'Yunhan Wang' """
    magk=np.arange(minmagk,maxmagk+deltamagk,deltamagk).T

    if magk.size == 0:
        return magk, []

    #Assume that any set of angular basis functions used in the program
    #form an orthonormal system on the surface of the sphere.

    #Require that the angular basis functions are the same in both models.
    #This can be relaxed, but more general cases are not implemented.
    if onevobjA.BasisFunctionType != onevobjB.BasisFunctionType:
        h.errPrint('get_FSC: onevobjA.BasisFunctionType %d ~= onevobjB.BasisFunctionType %d',
              onevobjA.BasisFunctionType,onevobjB.BasisFunctionType);

    R1A=onevobjA.R1
    R2A=onevobjA.R2
    llA=onevobjA.clnp.l
    mmA=onevobjA.clnp.n #Spherical harmonics notation of 'm' not icosahedral harmonics notation of 'n'.
    ppA=onevobjA.clnp.p
    cA=onevobjA.clnp.c

    R1B=onevobjB.R1
    R2B=onevobjB.R2
    llB=onevobjB.clnp.l
    mmB=onevobjB.clnp.n #Spherical harmonics notation of 'm' not icosahedral harmonics notation of 'n'.
    ppB=onevobjB.clnp.p
    cB=onevobjB.clnp.c

    #Cannot use the Htable that may be a part of either virusobj because
    #those values of $|\vec k|$ are unrelated to the values stored in
    #magk.    The following code is modified from
    #virusobj_set_2Dreciprocal.m .
    s_magk = size(magk)

    if R1A>=0.0:
        h.errPrint('get_FSC: Class A, only R1<0.0 (not R1>=0.0) is implemented. R1 %g',R1A)
    unique_lpA, map_lp2uniqueA, map_unique2lpA =unique2(zip(llA,ppA), "row")
    # TODO: below is really slow!! b/c using jroots_c
    HtableA=set_H0table_c(unique_lpA[:,0], unique_lpA[:,1], magk, np.zeros(s_magk), R2A)
    # HtableA = load('HtableA.mat','HtableA')[0]

    if R1B>=0.0:
        h.errPrint('get_FSC: Class B, only R1<0.0 (not R1>=0.0) is implemented. R1 %g',R1B)
    unique_lpB, map_lp2uniqueB, map_unique2lpB =unique2(zip(llB,ppB), 'row')
    # TODO: below is really slow!! b/c using jroots_c
    HtableB=set_H0table_c(unique_lpB[:,0], unique_lpB[:,1], magk, np.zeros(s_magk), R2B)
    # HtableB = load('HtableB.mat','HtableB')[0]

    SAB=np.zeros(s_magk)
    SAA=np.zeros(s_magk)
    SBB=np.zeros(s_magk)

    #Zhye Yin JSB 2003 Eq. 23.    The l,n,p sums are combined in the ii loop.
    # The p^\prime sum is the jj loop.

    #Compute SAA and SAB
    for ii in range(len(llA)):
        l=llA[ii]
        m=mmA[ii]
        Hlp=HtableA[:,map_unique2lpA[ii]]

        # Find the weights in the B class that have these l,m and any p^\prime
        indices=np.intersect1d(find_operators(llB, l, '='), find_operators(mmB, m, '='))
        # fprintf(1,'get_FSC: ii1 ii %d l %d m %d size(Hlp) %d %d size(indices from B) %d %d\n',ii,l,m,size(Hlp),size(indices));

        if indices.size != 0:
            for jj in indices:
                Hlpprime=HtableB[:,map_unique2lpB[jj]]
                # fprintf(1,'get_FSC: ii1-jj1 ii %d l %d m %d size(Hlp) %d %d jj %d size(Hlpprime) %d %d\n',ii,l,m,size(Hlp),jj,size(Hlpprime));
                SAB=SAB + format(cA[ii])*format(cB[jj])*format(Hlp)*format(Hlpprime)

        #Find the weights in the A class that have these l,m and any p^\prime
        indices=np.intersect1d(find_operators(llA, l, '='), find_operators(mmA, m, '='))

        if indices.size != 0:
            for jj in indices:
                Hlpprime=HtableA[:,map_unique2lpA[jj]]
                # fprintf(1,'get_FSC: ii1-jj2 ii %d l %d m %d size(Hlp) %d %d jj %d size(Hlpprime) %d %d\n',ii,l,m,size(Hlp),jj,size(Hlpprime));
                SAA=SAA + format(cA[ii])*format(cA[jj])*format(Hlp)*format(Hlpprime)
        # close for ii=1:length(llA)

    #Compute SBB
    for ii in range(len(llB)):
        l=llB[ii]
        m=mmB[ii]
        Hlp=HtableB[:,map_unique2lpB[ii]]
        # fprintf(1,'get_FSC: ii2 ii %d l %d m %d size(Hlp) %d %d\n',ii,l,m,size(Hlp));

        # Find the weights in the B class that have these l,m and any p^\prime
        indices=np.intersect1d(find_operators(llB, l, '='), find_operators(mmB, m, '='))
        if indices.size != 0:
            for jj in indices:
                Hlpprime=HtableB[:,map_unique2lpB[jj]]
                # fprintf(1,'get_FSC: ii2-jj1 ii %d l %d m %d size(Hlp) %d %d jj %d size(Hlpprime) %d %d\n',ii,l,m,size(Hlp),jj,size(Hlpprime));
                SBB=SBB + format(cB[ii])*format(cB[jj])*format(Hlp)*format(Hlpprime)
    fsc=np.hstack((SAB,SAA,SBB,SAB/np.sqrt(SAA*SBB)))
    return magk,fsc

def WriteMRC(map_data,filename):
    """
    Writing MRC is quite different from Matlab, it does not use rez variable
    """
    sio.savemat(filename,{'map_data':map_data})
    #map_data = map_data.transpose()
    #e = EMNumPy.numpy2em(map_data)
    #print(map_data)
    #e.write_image(filename)

######### for the linear group ########
def EM_expmax_iterate2convergence(vk,y,vobj,pixelnoisevar,EM_iter,all_tilde_b):
    sys.stdout.write('EM_expmax_iterate2convergence: size(vk) %d %d size(y) %d %d\n'%(vk.shape[0],vk.shape[1],y.shape[0],y.shape[1]));

    Neta=len(vobj);

    cbarNEW=np.zeros((Neta,1),dtype=object)
    nuNEW=np.zeros((Neta,1),dtype=object)
    qNEW=np.zeros((Neta,1))

    cbarTMP=np.zeros((Neta,1),dtype=object) # Software workaround so that is_clnp_converged(.) can be used.

    # Determine if the problem is all heterogeneous versus all homogeneous versus mixed.

    is_hetero=np.zeros((Neta,1),dtype=bool)
    for eta in xrange(0,Neta):
        if vobj[eta].nu.size == 0:
            """ ATTENTION I wonder if here I should use 0 and 1 instead of True and False"""
            is_hetero[eta] = False
        else:
            is_hetero[eta] = True

    if np.all(is_hetero==True):
        is_all_hetero=True
        sys.stdout.write('EM_expmax_iterate2convergence: all classes are heterogeneous\n')
    elif np.all(is_hetero==False):
        is_all_hetero=False;
        sys.stdout.write('EM_expmax_iterate2convergence: all classes are homogeneous\n')
    else:
        sys.exit('EM_expmax_iterate2convergence: mixed heterogeneous and homogeneous problem\n')

    # If the problem is all homogeneous, set all of the empty nu's to zero vectors.
    if not is_all_hetero:
        sys.stdout.write('EM_expmax_iterate2convergence: problem has only homogeneous classes, setting vobj{eta}.nu to 0 vector\n')
        for eta in xrange(0,Neta):
            vobj[eta].nu=np.zeros((vobj[eta].cbar.shape))

    # Print the initial conditions.
    sys.stdout.write('EM_expmax_iterate2convergence: Initial pixelnoisevar: %g\n'%pixelnoisevar)
    for eta in xrange(0,Neta):
        sys.stdout.write('EM_expmax_iterate2convergence: Initial cbar{%d} from vobj:\n'%eta)
        disp_v(EM_iter.verbosity,1,vobj[eta].cbar.reshape(-1,1))
        sys.stdout.write('EM_expmax_iterate2convergence: Initial nu{%d} from vobj:\n'%eta)
        disp_v(EM_iter.verbosity,1,vobj[eta].nu.reshape(-1,1))
        sys.stdout.write('EM_expmax_iterate2convergence: Initial q(%d) from vobj: %g\n'%(eta,vobj[eta].q))

    """
    Allocate and initialize history variables.  Except for loglike_history, the history
    variables preserve the state (cbar, nu, and q in vobj and pixelnoisevar) at the start
    of each iteration (so the values that were used to call this function are preserved)
    with the final entries being the state at the close of the terminal iteration.
    """
    cbar_history=np.zeros((Neta,EM_iter.maxiter+1),dtype=object)
    nu_history=np.zeros((Neta,EM_iter.maxiter+1),dtype=object)
    q_history=np.zeros((Neta,EM_iter.maxiter+1))
    pixelnoisevar_history=np.zeros((EM_iter.maxiter+1,1))
    loglike_history=np.zeros((EM_iter.maxiter+1,1))
    loglike_history.fill(-np.inf) # set to -Inf

    convergedtwice=False # require convergence to be achieved for two iterations in a row
    toomanyiterations=False # exited the loop not because convergence was achieved but because too many iterations were used.
    if is_all_hetero: # all classes are heterogenous models.
        # Previously assumed that the nu initial condition was not set and therefore executed havesetnuic=false
        # Now test if the initial condition on nu is already set <==> nu>0.0 for all classes and all elements.
        # If desire to set nu initial conditions here, can always read nu=0.0 from a file or set nu=0.0 when changing from a homogeneous to a heterogeneous model.
        havesetnuic=True
        for eta in xrange(0,Neta):
            if np.any(vobj[eta].nu<=0.0):
                havesetnuic=False
                break
    else: # all classes are homogenous models.
        havesetnuic=False

    ##################################################
    # Main loop over iterations starts here.
    ##################################################
    itr=0
    while True:
        sys.stdout.write('EM_expmax_iterate2convergence: Iteration %d\n'%itr)

        # Copy the current value of the state into the history variables.
        # Do nothing about loglike_history.
        for eta in xrange(0,Neta):
            cbar_history[eta][itr]=vobj[eta].cbar
            nu_history[eta][itr]=vobj[eta].nu
            q_history[eta][itr]=vobj[eta].q

        pixelnoisevar_history[itr]=pixelnoisevar
        # set loglike_history(itr) below.

        ##################################################
        # Beginning of joint update of pixelnoisevar, q, and cbar.
        ##################################################

        """
        Want to do a joint update of pixelnoisevar, q, and cbar.  So do not want to copy the new
        value of pixelnoisevar (which is contained in pixelnoisevarNEW) into pixelnoisevar until
        the new values of q and cbar are computed (which are contained in qNEW and cbarNEW).
        """
        pixelnoisevarNEW=pixelnoisevar
        if itr<EM_iter.maxiter4pixelnoisevarupdate and \
        (is_all_hetero or (not is_all_hetero and EM_iter.estimate_noise_var_in_homogeneous_problem)):
            # Update pixelnoisevar.

            sys.stdout.write('EM_expmax_iterate2convergence: itr %d compute p(nuisance|data,parameters) for update of pixelnoisevar\n'%itr)
            tstart=tm.time()
            # Compute $p(\theta_i,\eta_i|y_i,\bar{c}, V, q)$ by ZhengWangDoerschuk JOSA-A 2012 Eq. 22.
            (p_theta_eta,loglike2)=set_p_theta_eta_loglike_diagV(EM_iter.rule,vk,y,pixelnoisevar,vobj,all_tilde_b)
            sys.stdout.write('EM_expmax_iterate2convergence: itr %d (%g) done p(nuisance|data,parameters) for update of pixelnoisevar\n'%(itr,tm.time()-tstart))

            sys.stdout.write('EM_expmax_iterate2convergence: itr %d compute pixelnoisevarNEW (pixelnoisevar %g)\n'%(itr,pixelnoisevar))
            for eta in xrange(0,Neta):
                if not np.all(np.isfinite(p_theta_eta[eta][0])):
                    sys.exit('EM_expmax_iterate2convergence: ~isfinite(p_theta_eta{eta=%d}) is true\n'%eta)

            tstart=tm.time()
            # ZhengWangDoerschuk JOSA-A 2012 Section 4.B.3 p. 964.
            pixelnoisevarNEW=set_Q_noise_solve(EM_iter.rule,vk,y,vobj,p_theta_eta,EM_iter.Na, all_tilde_b);
            sys.stdout.write('EM_expmax_iterate2convergence: itr %d (%g) done pixelnoisevarNEW %g\n'%(itr,tm.time()-tstart,pixelnoisevarNEW))
            if not np.isfinite(pixelnoisevarNEW):
                sys.exit('EM_expmax_iterate2convergence: ~isfinite(pixelnoisevarNEW) is true\n')
        else:
            sys.stdout.write('EM_expmax_iterate2convergence: itr %d DID NOT compute p(nuisance|data,parameters) for update of pixelnoisevar\n'%itr)

        # Update cbar and q.
        sys.stdout.write('EM_expmax_iterate2convergence: itr %d update cbar and q\n'%itr)
        tstart=tm.time()
        """
        ZhengWangDoerschuk JOSA-A 2012 Eqs. 26 and 33.
        Updating cbar and q require $p(\theta_i,\eta_i|y_i,\bar{c}, V, q)$ and that is computed
        within EMstep_cbar_dV_large (by set_p_theta_eta_loglike_diagV) using the new value of the
        pixel noise covariance.
        """
        sys.stdout.write('EM_expmax_iterate2convergence: itr %d compute new cbar and q and loglike via EMstep_cbar_dV_large\n'%itr)
        """
        ATTENTION:
        I do not know why here can assign a value to a function
        """
        (cbarNEW,qNEW,loglike_history[itr,0],solvedlinearsystem)=EMstep_cbar_dV_large(EM_iter.rule,vk,y,pixelnoisevar,vobj,all_tilde_b,EM_iter.MinimumClassProb) # PASS all_tilde_b
        sys.stdout.write('EM_expmax_iterate2convergence: itr %d (%g) done new cbar and q and loglike via EMstep_cbar_dV_large\n'%(itr,tm.time() - tstart))
        sys.stdout.write('EM_expmax_iterate2convergence: itr %d done new loglikelihood %.16g\n'%(itr,loglike_history[itr,0]))


        # Copy the old cbar from vobj so that is_clnp_converged(.) can be used.
        for eta in xrange(0,Neta):
            cbarTMP[eta][0]=vobj[eta].cbar

        # Test if the loglikelihood has converged.
        absoluteloglike=itr>1 and \
          (EM_iter.loglikeftol<0.0 or abs(loglike_history[itr][0]-loglike_history[itr-1][0])<=EM_iter.loglikeftol)
        relativeloglike=itr>1 and \
          (EM_iter.loglikertol<0.0 or abs(loglike_history[itr][0]-loglike_history[itr-1][0])/(0.5*(abs(loglike_history[itr][0])+abs(loglike_history[itr-1][0])))<=EM_iter.loglikertol);
        loglike_converge=absoluteloglike and relativeloglike

        # Test if cbar has converged.
        (relativecbar,absolutecbar)=is_clnp_converged(cbarNEW,cbarTMP,EM_iter.cbarftol,EM_iter.cbarrtol,EM_iter.cbarftol_dividebyNc)
        sys.stdout.write('EM_expmax_iterate2convergence: itr %d relativecbar %g absolutecbar %g\n'%(itr,relativecbar,absolutecbar))
        cbar_converge=(EM_iter.cbarftol<0.0 or absolutecbar) and (EM_iter.cbarrtol<0.0 or relativecbar)

        # Update part of the state=(virusobj,pixelnoisevar).
        pixelnoisevar=pixelnoisevarNEW # This may do nothing if, in fact, pixelnoisevar was not updated.
        for eta in xrange(0,Neta):
            vobj[eta].clnp.c=cbarNEW[eta][0]
            vobj[eta].cbar=cbarNEW[eta][0]
            vobj[eta].q=qNEW[eta][0]

        """
        Combine the tests, and make sure that convergence has occurred for two updates in a row.
        Note that these two updates will be separated by an update of nu and q.
        Qiu Wang code uses || not && and uses an extremely strict criteria for loglike that is essentially never satisfied.
        """
        if cbar_converge and loglike_converge:
            sys.stdout.write('EM_expmax_iterate2convergence: The cbar estimation converged at least once (cbar_converge %d loglike_converge %d).\n'%(cbar_converge,loglike_converge))
            if convergedtwice:
                sys.stdout.write('EM_expmax_iterate2convergence: The cbar estimation has converged a second time in a row.\n')
                loglikefinal=loglike_history[itr][0]
                toomanyiterations=False
                break # break out of "while true" loop, only exit for convergence
            else:
                sys.stdout.write('EM_expmax_iterate2convergence: failed convergedtwice test\n')
                convergedtwice=True # save the fact that convergence occured once (but not twice)

        else: # neither cbar nor the loglikelihood converged.
            sys.stdout.write('EM_expmax_iterate2convergence: failed cbar_converge AND loglike_converge test\n')
            convergedtwice=False

        ##################################################
        # End of joint update of pixelnoisevar, q, and cbar.
        ##################################################

        if convergedtwice and is_all_hetero:

            ##################################################
            # Beginning of joint update of q and nu.
            ##################################################

            """
            While updating cbar and q, have converged once on cbar and/or loglikelihood, now update nu and q.
            Only do this if the problem has only heterogeneous classes.

            Always start the nu optimization from an initial condition that is
            EM_iter.nu_ic_FractionOfcbar times the current value of cbar.  The Qiu
            Wang code sets this initial condition from the current cbar every time
            nu is optimized.  In this code, that is made optional: either do as in
            the Qiu Wang code or in the first optimization of nu the initial
            condition is set from the current cbar but in later optimizations of
            nu the initial condition is the result from the immediately preceding
            optimization of nu.
            """

            sys.stdout.write('EM_expmax_iterate2convergence: itr %d about to update nu and q\n'%itr)

            if not havesetnuic and not EM_iter.nu_ic_always_proportional2cbar.size == 0:
                sys.stdout.write('EM_expmax_iterate2convergence: itr %d about to set nu initial conditional proportional to the current cbar\n'%itr)
                for eta in xrange(0,Neta):
                    if not havesetnuic or (not EM_iter.nu_ic_always_proportional2cbar.size == 0 and EM_iter.nu_ic_always_proportional2cbar(eta)):
                        if EM_iter.nu_ic_FractionOfcbar>=0.0:
                            vobj[eta].nu=mean2stddev(vobj[eta].cbar,EM_iter.nu_ic_FractionOfcbarForMinimum,EM_iter.nu_ic_FractionOfcbar)**2
                        else:
                            vobj[eta].nu=mean2stddev(vobj[eta].cbar,EM_iter.nu_ic_FractionOfcbarForMinimum,EM_iter.nu_ic_FractionOfcbar)

                havesetnuic=True
            else:
                #sys.stdout.write('EM_expmax_iterate2convergence: itr %d nu initial conditional not changed\n',itr);
                print("EM_expmax_iterate2convergence: itr %d nu initial conditional not changed")
                print(itr)

            for eta in xrange(0,Neta):
                #sys.stdout.write('EM_expmax_iterate2convergence: itr %d compute p(nuisance|data,parameters) for update of nu and q\n'%itr)
                tstart=tm.time()
                # Compute $p(\theta_i,\eta_i|y_i,\bar{c}, V, q)$ by ZhengWangDoerschuk JOSA-A 2012 Eq. 22 or equivalently
                # p223.yzheng.qiuwang.heterogeneous.GaussianMixture.tex 2011-07-31 Eq. 25.
                """
                ATTENTION
                I have no idea what does set_p_theta_eta_loglike_diagV returns, so I just leave this statement untreated
                """
                (p_theta_eta,loglike2)=set_p_theta_eta_loglike_diagV(EM_iter.rule,vk,y,pixelnoisevar,vobj, all_tilde_b)

                 #sys.stdout.write(1,'EM_expmax_iterate2convergence: itr %d (%g) done p(nuisance|data,parameters) for update of nu and q\n'%(itr,tm.time()-tstart))

                # Update q.
                WEIGHT_INDEX=6 # the index of the abscissa weights in the rule data structure
                sys.stdout.write('EM_expmax_iterate2convergence: itr %d compute qNEW\n'%itr)
                tstart=tm.time()
                # ZhengWangDoerschuk JOSA-A 2012 Eq. 26.
                Nv=y.shape[1]
                #qNEW[eta,0] = qNEW[eta,0].tolist()
                #qNEW[eta,0]=(p_theta_eta[eta][0]*EM_iter.rule[:,WEIGHT_INDEX-1]).sum(axis=0)/Nv;
                qNEW[eta, 0] = (p_theta_eta[eta][0] * EM_iter.rule[:, WEIGHT_INDEX - 1]).sum()/ Nv;
                #sys.stdout.write('EM_expmax_iterate2convergence: itr %d (%g) done qNEW(eta=%d) %g:\n'%(itr,tm.time()-tstart,eta,qNEW(eta)))

                # Set lower and upper bounds for the nu optimization problem.
                lb=np.zeros((vobj[eta].cbar.shape[1],1))
                ub=np.ones((vobj[eta].cbar.shape[1],1))*np.Inf

                # Update nu.
                #sys.stdout.write('EM_expmax_iterate2convergence: itr %d compute nuNEW using fmincon\n',itr);
                tstart=tm.time()
                # ZhengWangDoerschuk JOSA-A 2012 from Eq. 35 using Eqs. 43 and 44.
                """ATTENTION:
                I have no idea how to deal with fmincon
                """

                io.savemat('fmin.mat',mdict={'eta': eta, 'tilde_b': all_tilde_b[eta].tilde_b, 'vobj': vobj, 'EM_iter': EM_iter, 'pixelnoisevar': pixelnoisevar, 'vk': vk, 'y': y, 'p_theta_eta': p_theta_eta},long_field_names='true')
                nu = vobj[eta].nu
                nuNEW[eta] = minimize(objFun.setQem, nu, jac=True, hess=hess.setHQem, method='trust-ncg', options={'maxiter': 8}, tol=EM_iter.V_TolX)
                #nuNEW[eta] = minimize(objFun.setQem, vobj[eta].nu, jac=True, hess=hess.setHQem,
                #   method='trust-ncg', options={'maxiter':8, 'disp':True}, tol=EM_iter.V_TolX )

                """
                nuNEW[eta]=fmincon(@(nu) set_Q_dV_unc(EM_iter.rule,vk,y,pixelnoisevar,vobj,p_theta_eta,nu), \
                vobj[eta].nu,[],[],[],[],lb,ub,[],optimset('Algorithm','trust-region-reflective','GradObj','on', \
                'Hessian','on','TolX',EM_iter.V_TolX,'MaxIter',EM_iter.V_MaxIter,'Display','itr'))
                """
             #   sys.stdout.write('EM_expmax_iterate2convergence: itr %d (%g) done nuNEW(eta):\n'%(itr,tm.time()-tstart,eta))
                disp_v(EM_iter.verbosity,2,nuNEW[eta].reshape(-1,1))

                # Update part of the state=(virusobj,pixelnoisevar).
                vobj[eta].nu=nuNEW[eta]
                vobj[eta].q=qNEW[eta]

            #close 'for eta=1:Neta' loop

        ##################################################
        # End of joint update of q and nu.
        ##################################################

        else: # close 'if convergedtwice && is_all_hetero' test
            sys.stdout.write('EM_expmax_iterate2convergence: failed convergedtwice AND is_all_hetero test\n')



        # Test if the maximum number of iterations has been reached.  If yes, break.  If no, increment itr and continue.
        if itr>=EM_iter.maxiter:
            toomanyiterations=True
            break # break out of "while true" loop, only exit for NON convergence

        itr=itr+1;
    # close "while true" loop

    """
    Something is always done during an iteration of the loop and the state is immediately updated.
    So retain the last group of events by copying the state to the history variables one more time.
    It is possible to leave the loop via two 'break' statements: "convergence" or "too many iterations".
    In both cases, the itr-th value of the history variables has already been set at the beginning of
    the loop and the last group of events should be placed in the (itr+1)-th value.
    """

    for eta in xrange(0,Neta):
        """ ATTENTION
        Origin code is
        for eta=1:Neta
            cbar_history{eta,itr+1}=vobj{eta}.cbar;
            nu_history{eta,itr+1}=vobj{eta}.nu;
            q_history(eta,itr+1)=vobj{eta}.q;
        end
        I'm not quite sure with the index, and I changed itr start with 0
        """
        cbar_history[eta,itr+1]=vobj[eta].cbar
        nu_history[eta,itr+1]=vobj[eta].nu
        q_history[eta,itr+1]=vobj[eta].q

    pixelnoisevar_history[itr+1,0]=pixelnoisevar;
    # set loglike_history(itr+1) to -Inf above.

    if toomanyiterations:
        sys.stdout.write('EM_expmax_iterate2convergence: no convergence in maxiter %d iterations\n',EM_iter.maxiter)
        loglikefinal=loglike_history[itr][0]
    else:
        sys.stdout.write('EM_expmax_iterate2convergence: failed toomanyiterations test\n')


    if not is_all_hetero:
        # All classes are homogeneous classes.  Therefore, return to vobj{eta}.nu=[].
        sys.stdout.write('EM_expmax_iterate2convergence: problem has only homogeneous classes, returning vobj{eta}.nu to [] vector\n')
        for eta in xrange(0,Neta):
            if np.any(vobj[eta].nu!=0.0):
                sys.exit('EM_expmax_iterate2convergence: eta %d homogeneous problem but vobj{eta}.nu ~= 0 vector\n'%eta)

            vobj[eta].nu=np.array([])
    else:
        sys.stdout.write('EM_expmax_iterate2convergence: failed ~is_all_hetero test\n')


    if not (EM_iter.fn_savehistory == '' or EM_iter.fn_savehistory == None):
        sys.stdout.write('EM_expmax_iterate2convergence: about to save history and return\n')
        """ATTENTION
        Here it just ssave workspace, so I do nothing with it
        """
        # save(EM_iter.fn_savehistory,'cbar_history','nu_history','q_history','pixelnoisevar_history','loglike_history');
    else:
        sys.stdout.write('EM_expmax_iterate2convergence: about to not save history and return\n')

    return vobj,pixelnoisevar,loglikefinal,toomanyiterations


def fw(SNR,vobj,vkminimalset,Nv,NT,Na,Nb,rule,all_tilde_b,ixminimalset,iyminimalset):
    # Can do only one image per virus particle, no tilt series.
    if NT!=1:
        sys.exit('fw: NT %d\n'%NT)

    if vkminimalset.ndim!=2 or vkminimalset.shape[1]!=2:
        sys.exit('fw: dimensions of vkminimalset\n')

    WEIGHT_INDEX=6 # which column of rule(:,:) has the weights

    Neta=len(vobj)

    # Noise-free 2-D reciprocal-space images in the format expected by the estimation functions.
    y_nonoise=np.zeros((2*vkminimalset.shape[0],Nv))
    # Noisy 2-D real-space images.
    img=np.zeros((Nv,1),dtype=object)

    # Prepare for random selection of class.
    q_all=np.zeros((Neta,1))
    for eta in xrange(0,Neta):
        q_all[eta,0]=vobj[eta].q

    cumq=np.cumsum(q_all)
    assert cumq[cumq.size-1]==1 # the sum of all class porbabilities should equal to 1.

    # Prepare for random selection of projection direction and origin offset.
    # While it would be possible to use a class-specific rule, that is not currently implemented.
    cumtheta=np.cumsum(rule[:,WEIGHT_INDEX-1],0)/sum(rule[:,WEIGHT_INDEX-1],0);

    # Compute the Cholesky factorization of V (the weight covariance matrix).
    choleskyfactor=np.zeros((Neta,1),dtype=object)
    for eta in xrange(0,Neta):
        if vobj[eta].nu.size==0:
            choleskyfactor[eta][0]=np.array([])
        else:
            choleskyfactor[eta][0]=vobj[eta].nu**0.5

    # Compute 2-D reciprocal-space images by y=Lc.
    truevalues=np.zeros((Nv,2)) # truevalues(:,1)=true eta, truevalues(:,2)=true index into rule
    mean_all_pixels=0.0
    for ii in xrange(0,Nv):
        # Determine the realization of the class.
        eta = (cumq>=np.random.uniform(0,1,(1,1))[0][0]).nonzero()[0][0]
        truevalues[ii][0]=eta;
        # Determine the realization of the projection direction and origin offset.
        itheta=(cumtheta>=np.random.uniform(0,1,(1,1))[0][0]).nonzero()[0][0]
        truevalues[ii][1]=itheta;
        # Determine the realization of the coefficients.
        """
        ATTENTION: Here vobj[eta].clnp.c should be a N*1 matrix, however, it turns out to be a 1*N one, so I change shape[1] to shape[0]
        """
        Nc=vobj[eta].clnp.c.shape[0]
        c=vobj[eta].clnp.c.reshape(-1,1)

        if not vobj[eta].nu.size == 0:
            # After this statement, the result (matrix c) is slightly different from matlab,I think it's caused by standard_normal
            c=np.add(c,choleskyfactor[eta][0].reshape(-1,1)*np.random.standard_normal((Nc,1)))

        # Compute L
        Rabc=euler2R(rule[itheta,0:3])
        L=setL_nord(Rabc, vobj[eta].clnp.l, vobj[eta].clnp.n, vkminimalset, vobj[eta].Htable, vobj[eta].map_unique2lp, all_tilde_b[eta].tilde_b)
        # TODO though the below statement is correct, it's toooooooo slow!!!!
        # It checks if all element in a matrix is real
        """
        for i in np.nditer(L.T.copy(order='C')):
            if not np.isreal(i):
                sys.exit('fw: ii %d L is not real\n',ii)
        """
        # TODO: L are same as matlab, however, np.rank does not work as matlab
        """
        if np.rank(L) < Nc:
            sys.stderr.write('L is not full rank! ii %d rank(L) %d Nc %d.\n'%(ii,np.rank(L),Nc))
        """
        # Compute 2-D reciprocal-space images in the format expected by the estimation functions.
        result = np.dot(L,c)
        y_nonoise[:,ii]=result[:,0]
        # Compute 2-D real-space images as 2-D arrays.

        ImgAsRealVector=y_nonoise[:,ii]
        ImgAsComplexVector=(ImgAsRealVector[0:ImgAsRealVector.size:2]+complex(0,1)*ImgAsRealVector[1:ImgAsRealVector.size:2])

        ImgAs2DComplexImage=recip_Cvec2Cmat_conj_sym(ImgAsComplexVector,Na,Nb,ixminimalset,iyminimalset)
        """
        ATTENTION:
        In the below statement, though the image part of a complex num is 0,
        after cacluation it gets very small image number, usually e-15,
        and the real part is QUITE different from what matlab provide
        """
        img[ii][0]=(np.fft.fftshift(np.fft.ifft2(ImgAs2DComplexImage))).astype(np.float)
        mean_all_pixels=np.sum(np.sum(img[ii][0]))

    
    ###### modified by Weidan, the original one will overflow     
    #   mean_all_pixels=mean_all_pixels/(Nv*Na*Nb) 
    mean_all_pixels=mean_all_pixels/Nv
    mean_all_pixels=mean_all_pixels/Na
    mean_all_pixels=mean_all_pixels/Nb

    if SNR==np.Inf:
        sys.stdout.write('fw: SNR==Inf, will return noise-free 2-D real-space images\n')
        """ATTENTION
        Here in matlab it just returns before variable pixelnoisevar did something. So here I just return 0
        """
        return y_nonoise,img,truevalues,0

    # Compute covariance
    cov_all_pixels=0.0
    for ii in xrange(0,Nv):
        cov_all_pixels=cov_all_pixels+sum(sum((img[ii][0]-mean_all_pixels)**2))


    #### modified by Weidan
    #  cov_all_pixels=cov_all_pixels/(Nv*Na*Nb-1)
    temp=long(Nv)*long(Na)*long(Nb)-1
    cov_all_pixels=cov_all_pixels/(temp)

    # Compute the standard deviation of the noise to be added.
    # The is a SNR on standard deviation not on variance.
    stddev_of_noise=cov_all_pixels**0.5/SNR

    sys.stdout.write('fw: cov_all_pixels %g SNR %g stddev_of_noise %g\n'%(cov_all_pixels,SNR,stddev_of_noise))
    pixelnoisevar=stddev_of_noise**2

    sys.stdout.write('fw: pixelnoisevar %g\n'%pixelnoisevar)

    # Add noise.
    for ii in xrange(0,Nv):
        img[ii][0] += stddev_of_noise * np.random.normal(0, 1, (img[ii][0].shape[0], img[ii][0].shape[1]))
    return y_nonoise,img,truevalues,pixelnoisevar

def recip_Cvec2Cmat_conj_sym(Y2,Na,Nb,ixminimalset,iyminimalset):
    """
    @type Y2: a (m,1) vector

    @type Na, Nb is number
    """
    n=Y2.shape[0]

    Y3=np.NaN*np.ones((Na,Nb),dtype=np.cfloat)
    filled=np.zeros((Na,Nb))

    for m in xrange(0,n):
        # (x1,y1) was computed, (x2,y2) is the conjugate symmetric point
        """ATTENTION
        I did not find ixminimalset or iyminimalset
        """
        x1=ixminimalset[m][0]-1
        x2=conj_sym_index_1D(x1,Na-1) # Converting from matlab idx to python idx
        y1=iyminimalset[m][0]-1
        y2=conj_sym_index_1D(y1,Nb-1) # Converting from matlba idx to python idx

        if filled[x1][y1]==1 or filled[x2][y2]==1:
            sys.exit('m '+str(m))

        Y3[x1][y1]=Y2[m]

        if x1!=x2 or y1!=y2: # in case there is a small imaginary part
            Y3[x2][y2]=Y2[m].conj()
        elif np.imag(Y3[x1][y1])!=0.0:
            sys.stdout.write('%s: m %d (%d,%d) is self conjugate symmetric but has value %g+i*%g\n'%(inspect.getfile(inspect.currentframe()),m,x1,y1,np.real(Y3[x1][y1]),np.imag(Y3[x1][y1])))
            Y3[x1][y1]=np.real(Y3[x1][y1])

        filled[x1][y1]=1
        filled[x2][y2]=1


    minfilled=np.min(filled[:])
    maxfilled=np.max(filled[:])
    if minfilled!=1 or maxfilled != 1:
        sys.stdout.write('%s: min(filled(:)) %g max(filled(:)) %g\n'%(inspect.getfile(inspect.currentframe()),minfilled,maxfilled))


    (minerr,maxerr)=is_conj_sym(Y3)
    if minerr!=0.0 or maxerr!=0.0:
        sys.stdout.write('%s: Y3 conjugate symmetry: minerr %g maxerr %g\n'%(inspect.getfile(inspect.currentframe()),minerr,maxerr))

    return Y3

def is_conj_sym(R):
    ndim1=R.shape[0]
    ndim2=R.shape[1]
    testval=np.zeros((R.shape[0],R.shape[1]))
    for x1 in xrange(0,ndim1): # index into Matlab matrix
        x2=conj_sym_index_1D(x1,ndim1-1)
        for y1 in xrange(0,ndim2): # index into Matlab matrix
            y2=conj_sym_index_1D(y1,ndim2-1)
            if x1!=x2 or y1!=y2: # not going to count a possible imaginary part as error
                err=R[x1][y1]-R[x2][y2].conj()
            else:
                if np.imag(R[x1][y1])!=0.0:
                    sys.stdout.write('(%d,%d) is self conjugate symmetric but has value %g+i*%g\n'%(x1,y1,np.real(R[x1][y1]),np.imag(R[x1][y1])))

                err=0.0

            denominator=np.abs(R[x1][y1])+np.abs(R[x2][y2])
            if denominator!=0.0:
                testval[x1][y1]=np.abs(err)/denominator
            elif np.abs(err)==0.0:
                testval[x1][y1]=0.0
            else:
                testval[x1][y1]=np.abs(err)/denominator # really are dividing by 0



    minerr=np.min(testval[:])
    maxerr=np.max(testval[:])
    return minerr,maxerr

def disp_v(verbosity,vlevel,x):
    """
    @type x: a vector, vertically
    """
    if verbosity>=vlevel:
        print x

def mpi_set_p():
#	mat = io.loadmat('mpi_set_p.mat' % cnt)
	comm = MPI.COMM_SELF.Spawn(sys.executable, args=['funcs.py'],maxprocs=2)
	size = comm.Get_size()
	rank = comm.Get_rank()
	arr = np.zeros((4,4))
	if rank == 0:
		comm.Recv(arr, source=1)
		print arr
	else:
		arr[0][0] = 1
		comm.Send(arr, dest=0)
	return	
            #    (p_theta_eta,loglike2)=set_p_theta_eta_loglike_diagV(EM_iter.rule,vk,y,pixelnoisevar,vobj) 
	if rank == 0:
		set_p_theta_eta_loglike_diagV(mat['EM_iter']['rule'], mat['vk'], mat['y'], mat['pixelnoisevar'], mat['vobj'])
	else:
		while (True):
			data = comm.recv(source=0)
			L = setL_nord(data['Rabc'], data['vo.clnp.l'], data['vo.clnp.n'], data['vk'], data['vo.Htable'],
				data['vo.map_unique2lp'], data['tilde_b'])
			comm.send(L, dest=0, tag=data['n'])
	
	

def set_p_theta_eta_loglike_diagV(rule, vk, y, noise_var, virusobjs,all_tilde_b):
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
    t_p=tm.time()

    WEIGHT_INDEX = 5 # the index of the abscissa weights in the rule data structure

    vo = virusobjs # use a shorter name inside the function

    """ initialize variables """
    Neta = len(vo)
    # Ny = size(y, 1);
    Ni = y.shape[1]
    Nzi = rule.shape[0]
    exponents = np.zeros((Neta, 1),dtype=object)


    """ compute the scaling factor of the exponents """

    for eta in xrange(0, Neta):
        Nc = vo[eta].nu.shape[0]
        exponents_eta = np.zeros((Ni, Nzi))
        aaa = np.array(vo[eta].nu).flatten()
        if all(vo[eta].nu > 0.0):
            lndetV = np.sum(np.log(aaa))
        else:
            lndetV = 0

        # read tilde b
        #b_fn = 'callti.out.read_by_C'  # file contains \tilde{b} values computed from mathematica
        #lmax = np.max(vo[eta].clnp.l)
        #tilde_b = rd_b(b_fn, lmax)

        """ATTENTION
        I change parfor to for
        """

        for n in xrange(0, Nzi):
            Rabc = euler2R(rule[n, 0:3])
            L = setL_nord(Rabc, vo[eta].clnp.l, vo[eta].clnp.n, vk, vo[eta].Htable, vo[eta].map_unique2lp, all_tilde_b[eta].tilde_b)
            # L = jobs[n]()
            lndet_Sigma = lndet_fast(noise_var, L, vo[eta].nu,
                                     lndetV)  # this function has a relatively small difference with matlab, this is where the difference comes
            Lc = np.dot(L, vo[eta].cbar)

            y_Lc = y - Lc
            y_Lc_Sigma = y_Lc_Sigma_fast(noise_var, L, vo[eta].nu, y_Lc)  # use fast algorithm
            assert np.all(np.isfinite(
                y_Lc_Sigma)), 'set_p_theta_eta_loglike_diagV: assert(all(isfinite(y_Lc_Sigma))): eta %d n %d\n' % (eta, n)

            exponents_eta[:, n] = -0.5 * (y_Lc_Sigma[:] + lndet_Sigma)

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
        sys.exit('size of maxexp %d, size of minexp %d, shhould equal to Ni %d!\n'% (maxexp.shape[0], minexp.shape[0], Ni))

    expshift = np.zeros((Ni,1)) # vector of size Ni
    for img in xrange(0,Ni):
        expshift[img][0]=scaling_rule(minexp[img][0], maxexp[img][0])


    """ compute $$p(\theta_i,\eta_i | y_i, \bar{c}, V, q}$$ """
    sys.stdout.write('computing p...\n')
    Zeta=np.zeros((Neta, 1),dtype=object) # numerator $$p(y_i|\theta_i,\eta_i,\bar{c}^{\eta_i},V_{\eta_i})q_{\eta_i}$$
    Zetabar=np.zeros((Ni,1)) # denominator
    p=np.zeros((Neta, 1),dtype=object)

    for eta in xrange(0,Neta):
        Zeta[eta][0] = np.zeros((Ni, Nzi))
        for n in xrange(0,Nzi):

            #print("test")
            #print(np.exp(exponents[eta][0][:,n]+expshift[:].flat))


            Zeta[eta][0][:,n] = np.exp(exponents[eta][0][:,n]+expshift[:].flat)*vo[eta].q
            Zetabar[:] = Zetabar[:] + (Zeta[eta][0][:,n]*rule[n,WEIGHT_INDEX]).reshape(-1,1)


    assert np.all(Zetabar[:]!=0.0),'set_p_theta_eta_loglike_diagV: assert(all(Zetabar(:)~=0.0))'

    for eta in xrange(0,Neta):
        p[eta][0] = Zeta[eta][0]/ Zetabar


    """ compute the log likelihood """
    sys.stdout.write('computing the log likelihood...\n')
    loglike = np.sum(np.log(Zetabar)) - np.sum(expshift) # need to undo the effect of shifting exponents
    # fprintf(1, 'set_p_theta_eta_nu_noL: log likelihood %.16g.\n', loglike);

    sys.stdout.write('set_p_theta_eta_loglike_diagV time: %d\n'%(tm.time()-t_p))
    return p, loglike

def set_Q_noise_solve(rule, vk, y,virusobjs, p_theta_eta,Na,all_tilde_b):
    """
    function to compute Q as a function of noise variance
    read tilde_b in only once
    """
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
    #b_fn = 'callti.out.read_by_C' # file contains \tilde{b} values computed from mathematica
    #lmax = np.max(vo[eta].clnp.l)
    #tilde_b = rd_b(b_fn, lmax)

    for n in xrange(0,Nzi):
        Rabc = euler2R(rule[n, 0:3])
        L=setL_nord(Rabc, vo[eta].clnp.l, vo[eta].clnp.n, vk, vo[eta].Htable, vo[eta].map_unique2lp,all_tilde_b[eta].tilde_b)
        """ Use fast lndet algorithm """
        Lc = np.dot(L,vo[eta].cbar) # \mu = L*c;
        y_Lc =  y - Lc
        wri = np.dot(rule[n,WEIGHT_INDEX],p_theta_eta[eta][0][:,n])

        #print(rule[n,WEIGHT_INDEX])
        #print(wri)

        A=A+np.sum(wri)

        B=B+np.dot(np.sum(y_Lc**2,axis=0),wri)

    A=A*Na*Na
    #test
    #print(Na)
    noise_var=B/A

    sys.stdout.write('set_Q_noise_solve time: %d\n'%(tm.time()-t_Q))
    return noise_var

def y_Lc_Sigma_fast(q,L,nu,y_Lc):
    """
    @type: np.array
    @param L

    @type: np.array
    @param nu

    @type: number
    @param q
    """
    Nc=nu.size

    if q == 0:
        Sigma = np.dot(L,np.dot(np.diag(nu),L.T))
        # \ stands for left matrix division
        y_Lc_Sigma=np.sum(y_Lc  * (np.linalg.solve(Sigma,y_Lc)),axis=0)
        return y_Lc_Sigma
    elif q < 0:
        sys.exit('Error: noise_var not positive ')
        return -1

    y_Lc_Sigma = np.sum(1./q*(y_Lc**2),axis=0)
    if ty_nnz.ty_nnz(nu) == Nc:
        midmx=1/q*np.dot(L.T,L)
        for i in xrange(0,Nc):
            midmx[i][i]=midmx[i][i]+1./nu[i][0]
        LTy=np.dot(L.T,y_Lc)
        y_Lc_Sigma=y_Lc_Sigma-q**(-2)*np.sum(LTy*(np.linalg.solve(midmx,LTy)),axis=0)
    return y_Lc_Sigma

def lndet(A):
    """
    pivoting in LU factorization may result in negative diagonal entries.
    so we need to take the absolute value of the diagnal.
    """
    """
    ATTENTION
    I do not find lu in python
    """
    (AP, AL, AU) = scipy.linalg.lu(A)
    Alndet = np.sum(np.log(np.abs(np.diag(AU))))
    return Alndet

def lndet_fast(q,L,nu,lndetV):
    Nc=nu.shape[0]
    Ny=L.shape[0]

    if q==0.0:
        lndetSigma=lndet(np.dot(L,np.dot(np.diag(nu),L.T)))
        assert np.isfinite(lndetSigma),'Assertion lndet_fast #1'
        return lndetSigma
    elif q<0.0:
        sys.exit('lndet_fast: q %g is <0.0\n'%q)

    if all(nu>0.0):
        # standard case
        M=(np.dot(L.T,L))/q
        for i in xrange(0,Nc):
            M[i,i]=M[i,i]+1.0/nu[i][0]

        lndetSigma=Ny*math.log(q)+lndetV+lndet(M)
    else:
        # assume all(nu==0.0) is true and therefore drop the L V L^T term and return \ln\det(Q)
        lndetSigma=Ny*math.log(q)

    assert np.isfinite(lndetSigma), 'Assertion lndet_fast #2'
    return lndetSigma

def EMstep_cbar_dV_large(rule, vk, y, noise_var, virusobjs,all_tilde_b,MinimumClassProb):
    """
    find cbar for the diagonal V problem
    read tilde_b once

    t_p=clock
    """

    sys.stdout.write('This is function EMstep_cbar_dV_large\n')
    WEIGHT_INDEX = 5 # the index of the abscissa weights in the rule data structure

    vo = virusobjs # use a shorter name inside the function

    """ initialize variables """
    # Neta = vo.shape[0]
    Neta = len(vo)
    # Ny = size(y, 1);
    Nv = y.shape[1]
    Nzi = rule.shape[0]

    # solvedlinearsystem(1)=1 if solved F\g, =0 if used previous answer
    solvedlinearsystem=np.ones((Neta,1))
    # allocating storage
    cbarnew = np.zeros((Neta,1),dtype=object)
    qnew = np.zeros((Neta, 1))


    # compute $$p(\theta_i,\eta_i|y_i,\bar{c}, V, q)$$ by Eq. (25)
    sys.stdout.write('computing p_theta_eta...\n')
    (p_theta_eta, loglike)=set_p_theta_eta_loglike_diagV(rule, vk, y, noise_var, vo, all_tilde_b)
    # fprintf(1,'initial loglike = %f\n',loglike);
    n_nonzero_p_theta_eta=np.zeros((Neta,1))
    n_total_p_theta_eta=np.zeros((Neta,1))
    for eta in range(Neta):
        n_nonzero_p_theta_eta[eta][0]=np.sum((p_theta_eta[eta][0]>0.0).astype(int))
        n_total_p_theta_eta[eta][0]=np.prod(p_theta_eta[eta][0].shape)
    sys.stdout.write('EMstep_cbar_dV_large: n_total_p_theta_eta:');
    sys.stdout.write(' %g %g'%(n_total_p_theta_eta[0][0],n_total_p_theta_eta[1][0]))
    sys.stdout.write(' n_nonzero_p_theta_eta:')
    sys.stdout.write(' %g %g\n'%(n_nonzero_p_theta_eta[0][0],n_nonzero_p_theta_eta[1][0]))
    ## compute the expectation and maximize it to get new $$\bar{c}$$ and $$V$$
    sys.stdout.write('computing the expectation and maximize it... \n')
    for eta in xrange(0,Neta):
        Nc = vo[eta].cbar.shape[0]
        F = np.zeros((Nc, Nc))
        g = np.zeros((Nc, 1))
        qnew_eta = 0
        # compute the expectation with numerical integrations

        """
        # read tilde b
        b_fn = 'callti.out.read_by_C' # file contains \tilde{b} values computed from mathematica
        lmax = max(vo[eta].clnp.il);
        tilde_b = rd_b(b_fn, lmax);
        """
        for n in xrange(0,Nzi):

            Rabc = euler2R(rule[n,0:3]);
            L=setL_nord(Rabc, vo[eta].clnp.l, vo[eta].clnp.n, vk, vo[eta].Htable, vo[eta].map_unique2lp,all_tilde_b[eta].tilde_b)

            (LT_SigmaInv_y,LT_SigmaInv_L)=LT_SigmaInv_fast(noise_var,L,vo[eta].nu,y)

            wri = rule[n,WEIGHT_INDEX]*p_theta_eta[eta][0][:,n]
            # update F and g

            g = g + np.dot(LT_SigmaInv_y,wri.reshape(-1,1)) # vector of size Nc
            sum_wri = np.sum(wri)
            F = F + np.dot(LT_SigmaInv_L,sum_wri) # matrix of size Nc-by-Nc
            qnew_eta = qnew_eta + sum_wri

        qnew[eta][0] = qnew_eta
        # maximization step
        MaxCondNumber4F=1.0e8;
        if np.linalg.cond(F)<MaxCondNumber4F:
            # do the update
            cbarnew[eta][0]=scipy.linalg.solve(F,g)
        else:
            # do not do the update
            if np.any((vo[eta].clnp.c-vo[eta].cbar)!=0.0): # see assert_UpdateBothCopiesOfcbar.m
                sys.stderr.write('EMstep_cbar_dV_large: cond(F) large and .clnp.c~=.cbar\n')
            cbarnew[eta][0]=vo[eta].clnp.c
            solvedlinearsystem[eta][0]=0



    # update the class probabilites
    qnew = qnew / Nv
    qnew=applyclassproblimit(qnew,MinimumClassProb)

    return cbarnew, qnew, loglike, solvedlinearsystem

def applyclassproblimit(q,MinimumClassProb):
    if q.size==1 or MinimumClassProb==0.0:
        qNEW=q
        return qNEW

    toosmallindices=(q<MinimumClassProb).nonzero()

    if toosmallindices[0].size==0:
        sys.stdout.write('applyclassproblimit: no change\n')
        qNEW=q
        return qNEW

    sumoftoosmall=np.sum(q[toosmallindices[0]]);
    deltaprob=toosmallindices[0].size*MinimumClassProb - sumoftoosmall
    amount2subtract=deltaprob/(q.size-toosmallindices[0].size)

    sys.stdout.write('applyclassproblimit: q:')
    #sys.stdout.write(' %g\n'%q)

    bigenoughindices=(q>=MinimumClassProb).nonzero()

    qNEW=np.zeros(q.shape)
    qNEW[toosmallindices[0]]=MinimumClassProb
    for i in bigenoughindices[0]:
        qNEW[i] = q[i]-amount2subtract

    sys.stdout.write('applyclassproblimit: MinimumClassProb %g qNEW:'%MinimumClassProb)
    #sys.stdout.write(' %g\n'%qNEW)

    if (qNEW<0.0).nonzero()[0].size!=0:
        sys.stderr.write('applyclassproblimit: qNEW has negative values\n')
    """
    ATTENTION:
        #TODO I'm unable to find python equivalent of eps
    if np.sum(qNEW)-1.0>2.0*q.size*eps:
        sys.stderr.write('applyclassproblimit: qNEW not normalized\n');
    """
    return qNEW

def is_clnp_converged(cbarnew,cbar,ftol,rtol,ftol_dividebyNc):
    """
    This function applies the convergence criteria to each class separately and requires that all classes converge.
    Alternatively, the cbar values for all classes can be combined into one vector and the convergence criteria can be applied to that vector.
    """
    rconverged=True
    fconverged=True

    for eta in xrange(0,cbar.shape[0]):
        errorsignal=LA.norm(cbarnew[eta][0]-cbar[eta][0],1)
        averagesignal=0.5*(LA.norm(cbarnew[eta][0],1)+LA.norm(cbar[eta][0],1))
        if ftol_dividebyNc:
            # Has no effect on the rtol criteria.
            errorsignal=errorsignal/cbar[eta][0].shape[0]
            averagesignal=averagesignal/cbar[eta][0].shape[0]

        fconverged=(fconverged and (errorsignal<=ftol))
        rconverged=(rconverged and ((errorsignal/averagesignal)<=rtol))

    return rconverged,fconverged

def LT_SigmaInv_fast(q,L,nu,y):

    Nc=nu.shape[0]

    if q==0:
        Sigma=np.dot(L,np.dot(np.diag(nu),L.T))
        LT_SigmaInv_y=np.dot((L.T/Sigma),y)
        LT_SigmaInv_L=np.dot((L.T/Sigma),L)
        return LT_SigmaInv_y, LT_SigmaInv_L
    elif q<0:
        sys.exit('Error: noise_var not positive \n')

    LTy=np.dot(L.T,y)
    LTL=np.dot(L.T,L)
    LT_SigmaInv_y=1./q*LTy
    LT_SigmaInv_L=1./q*LTL
    if ty_nnz.ty_nnz(nu)==Nc:
        midmx=1./q*(np.dot(L.T,L))
        for i in xrange(0,Nc):
            midmx[i,i]=midmx[i,i]+1/nu[i]

        #test start
        #x = np.dot((LTL/midmx),LTy)
        #print(x)
        #test end

        LT_SigmaInv_y=LT_SigmaInv_y - q**(-2) * np.dot((LTL/midmx),LTy)
        LT_SigmaInv_L=LT_SigmaInv_L - q**(-2) * np.dot(LTL,np.linalg.solve(midmx,LTL))
        #print(LT_SigmaInv_y-q**(-2))
        #LT_SigmaInv_y=np.dot(np.dot(LT_SigmaInv_y-q**(-2), (LTL/midmx)),LTy)
        #LT_SigmaInv_L=np.dot(np.dot(LT_SigmaInv_L-q**(-2), LTL),np.linalg.solve(midmx,LTL))


    return LT_SigmaInv_y, LT_SigmaInv_L
'''
def ReadMRC(filename,startSlice=1, numSlices=np.Inf,test=0):
    """ATTENTION
    ReadMRC is quite different from Matlab Version, I do not use 'test' here,
    and here I only deal with 2-D case
    """
    e = EMData()
    e.read_image(filename)
    attr_dict = e.get_attr_dict()
    s={}
    s['nx']=attr_dict['nx']
    s['ny']=attr_dict['ny']
    s['nz']=min(attr_dict['nz']-(startSlice-1),numSlices)
    ma=attr_dict['maximum']
    mi=attr_dict['minimum']
    av=attr_dict['mean_nonzero']
    map_data = EMNumPy.em2numpy(e)
    map_data = map_data.transpose()
    map_data = map_data[:,:,startSlice-1:numSlices]
    return map_data,s,mi,ma,av
'''
def set_kbiggest(N,deltachi):
    """
    N is the number of samples.

    The DSP tradition would be to index 0:N-1.
    deltachi is the sampling interval.
    """
    if np.fix(N)!=N:
        sys.exit('set_kbiggest: N %g fix(N) %d\n'%(N,np.fix(N)))

    if N<=0:
        sys.exit('set_kbiggest: N %d <= 0\n'%N)

    if deltachi<=0.0:
        sys.exit('set_kbiggest: deltachi %g <= 0\n',deltachi);

    if np.fix(N/2)!=N/2:
        # N is odd
        nbiggest=(N-1)/2
    else:
        # N is even
        nbiggest=N/2


    kbiggest=nbiggest/(N*deltachi)
    return kbiggest

def scaling_rule(minexp, maxexp):
    """
    This is the Python version of scaling_rule.c
    Example: expshit = scaling_rule(min(exponent(:), max(exponent(:))
    """
    SAFETYFRAC = 0.9
    """
    if (minexp.shape != (1,1) or maxexp.shape != (1,1)):
        sys.exit('Either minexp or maxexp is not a scalar!\n')
    """
    # find out the range of exponent that is allowed
    minallow = math.log(sys.float_info.min)
    maxallow = math.log(sys.float_info.max)

    """
    Since the program will do exp(+exponent)*mess summed over many
    terms, I am somewhat concerned about overflow even if the exp
    itself is ok.  So I won't use the entire range of possible
    exponents -- I'd rather let more cases than necessary result in
    underflows.  In particular, I will only use
    [minallow,maxallow*SAFETYFRAC].
    """
    maxallow = maxallow*SAFETYFRAC
    tmp = (maxallow-minallow) - (maxexp-minexp)
    if tmp > 0.0:
        """
        The entire range of exponents will fit, get it centered inthe range
        #ifdef PRTINRANGE
        fprintf(2, 'inrange %g %g %g %g\n', minallow, maxallow, minexp, maxexp);
        #endif
        """
        expshift = minallow+tmp/2.0-minexp
    else:
        """
        {
          The entire range of exponents will not fit, put the largest at
          the top of the range
        }
        fprintf(2, 'outrange %g %g %g %g\n', minallow, maxallow, minexp, maxexp);
        """
        expshift = maxallow - maxexp

    return expshift
if __name__=='__main__':
    """ test cases """
    #rd_b("callti.out.read_by_C", 10)
    #(l1,l2)= load("FHV.out.lmax10pmax5.Neta2.rule49.50.imagestack.mat", 'imagestack','imageindex')
    #np.asarray([i-1 for i in l2])

    #shape = (91,91)
    #deltachi = np.asarray([4.7,4.7])
    #radiuscorner = np.asarray([202.1000,244.4000])
    #box_normalize_mask(shape, deltachi, radiuscorner)
    #load('FHV.out.lmax10pmax5.Neta2.rule49.50.inv.vobj.mat','vobj')
    #sphbes(0,1.57079632679)
    mpi_set_p()	
    jroots_c(0,5,10000)
