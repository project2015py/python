#!/usr/bin/env python
# structures.py
# Yunhan Wang (yw559)
# Dec 14, 2013
"""
all data structures for the FHV projects
"""

import numpy as np
from scipy import io
import sys
import os
from time import strftime as time

class Log():
    """ print to console, as well as to fn """
    def __init__(self, fn=None):
        self.terminal = sys.stdout
        if fn != None:
            self.toFile(fn)
        else:
            self.log_path = ''
        sys.stdout = self
        
    def write(self, message):
        self.terminal.write(message)
        if(len(self.log_path) > 0):
            log = open( self.log_path, 'a' )
            log.write( message )
            log.close()
    
    def toFile(self, fn):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.log_path = os.path.join(current_dir,fn)
        log = open( self.log_path, 'a' )
        log.write('\n########## %s ##########\n'%(time("%Y-%m-%d %H:%M:%S")))
        log.close()
        
class MatList(list):
    """ imitate matlab cell array, build a list of commands (mat_struct).
    an element in the list has fields operator, fn_diary, etc. """
    
    def __init__(self, fp):
        """ take in a mat file which has commands, construct command list """
        f = io.loadmat(fp, squeeze_me=True, struct_as_record=False) # what if empty
        for i in range(len(f['cmd'])):
            self.append(f['cmd'][i])


class Recipobj(object):
    """ the structure for recipobj """
    def __init__(self, ny, ii, vk, ix, iy, vkm):
        self.Ny4minimalset=ny
        self.iiminimalset=ii
        self.vkminimalset=vk
        self.ixminimalset=ix
        self.iyminimalset=iy
        self.vkmag=vkm

class SingleVirus():
    def init(self):
        self.clnp_fn=[] #  means private, hidden from the outside
        self.nu_fn, self.q_fn=[],[]
        self.clnp=Clnp()
        self.cbar, self.nu=[], []
        self.BasisFunctionType=0
        self.R1, self.R2, self.q=0,0,0
    
    def typecheck(self, name):
        """ for the linear group: add assertions for each type"""
        if name == "clnp_fn":
            assert True
        elif name == "nu_fn":
            assert True
        elif name == "q_fn":
            assert True
        elif name == "clnp":
            assert True
        elif name == "cbar":
            assert True
        elif name == "nu":
            assert True
        elif name == "BasisFunctionType":
            assert True
        elif name == "R1":
            assert True
        elif name == "R2":
            assert True
        elif name == "q":
            assert True
    
    def toString(self):
        return self.nu, self.R1, self.R2
        
class Clnp():
    def init(self):
        self.c=np.ndarray(shape=(15,1), dtype=float)
        self.l=np.ndarray(shape=(15,1), dtype=float)
        self.n=np.ndarray(shape=(15,1), dtype=float)
        self.p=np.ndarray(shape=(15,1), dtype=float)
        self.optflag=[] #np.ndarray(shape=(1,1), dtype=float)
    
#class SingleVirus():
#    def __init__(self):
#        self.__clnp_fn=[] # __ means private, hidden from the outside
#        self.__nu_fn, self.__q_fn=[],[]
#        self.__clnp=Clnp()
#        self.__cbar, self.__nu=[], []
#        self.__BasisFunctionType=0
#        self.__R1, self.__R2, self.__q=0,0,0
#    
#    def typecheck(self, name):
#        """ for the linear group: add assertions for each type"""
#        if name == "clnp_fn":
#            assert True
#        elif name == "nu_fn":
#            assert True
#        elif name == "q_fn":
#            assert True
#        elif name == "clnp":
#            assert True
#        elif name == "cbar":
#            assert True
#        elif name == "nu":
#            assert True
#        elif name == "BasisFunctionType":
#            assert True
#        elif name == "R1":
#            assert True
#        elif name == "R2":
#            assert True
#        elif name == "q":
#            assert True
#            
#    def set_clnp_fn(self, value):
#        self.__clnp_fn = value
#    def clnp_fn(self):
#        return self.__clnp_fn
#    
#    def set_nu_fn(self, value):
#        self.__nu_fn = value
#    def nu_fn(self):
#        return self.__nu_fn
#    
#    def set_q_fn(self, value):
#        self.__q_fn = value
#    def q_fn(self):
#        return self.__q_fn
#    
#    def set_clnp(self, value):
#        self.__clnp = value
#    def clnp(self):
#        return self.__clnp
#    
#    def set_cbar(self, value):
#        self.__cbar = value
#    def cbar(self):
#        return self.__cbar
#
#    def set_BasisFunctionType(self, value):
#        self.__BasisFunctionType = value
#    def BasisFunctionType(self):
#        return self.__BasisFunctionType
#    
#    def set_R1(self, value):
#        self.__R1 = value
#    def R1(self):
#        return self.__R1
#    
#    def set_R2(self, value):
#        self.__R2 = value
#    def R2(self):
#        return self.__R2
#    
#    def set_nu(self, value):
#        self.__nu = value
#    def nu(self):
#        return self.__nu
#    
#    def set_q(self, value):
#        self.__q = value
#    def q(self):
#        return self.__q
#    
#    def toString(self):
#        return self.nu, self.R1, self.R2
#        
#class Clnp():
#    def __init__(self):
#        self.__c=np.ndarray(shape=(15,1), dtype=float)
#        self.__l=np.ndarray(shape=(15,1), dtype=float)
#        self.__n=np.ndarray(shape=(15,1), dtype=float)
#        self.__p=np.ndarray(shape=(15,1), dtype=float)
#        self.__optflag=[] #np.ndarray(shape=(1,1), dtype=float)
#    def set_c(self, value):
#        self.__c = value
#    def c(self):
#        return self.__c
#    
#    def set_l(self, value):
#        self.__l = value
#    def l(self):
#        return self.__l
#
#    def set_n(self, value):
#        self.__n = value
#    def n(self):
#        return self.__n
#    
#    def set_p(self, value):
#        self.__p = value
#    def p(self):
#        return self.__p
#
#    def set_optflag(self, value):
#        self.__optflag = value
#    def optflag(self):
#        return self.__optflag

class EMMC():
    def __init__(self):
        self.Nrandic = 0.0
        self.FractionOfMeanForMinimum = 0.0
        self.FractionOfMean = 0.0
        
class EMIter():
    def __init__(self):
        self.maxiter = 0.0
        self.maxiter4pixelnoisevarupdate = 0.0
        self.cbarftol = 0.0
        self.cbarrtol =0.0
        self.cbarftol_dividebyNc =0.0
        self.loglikeftol =0.0
        self.loglikertol =0.0
        self.frac_of_cbar4nu_ic =0.0
        self.estimate_noise_var_in_homogeneous_problem =0.0
        self.nu_ic_always_proportional2cbar = np.ndarray(shape=(15,1), dtype=float)
        self.rule = np.ndarray(shape=(15,1), dtype=float)
        self.Na =0
        self.V_TolX =None
        self.V_MaxIter = None

class RealSpaceCubes():
    def __init__(self, cmd):
        self.whichclass = cmd.whichclass # ???? this is the index!
        self.wantrhobar = cmd.wantrhobar
        self.wantrxx = cmd.wantrxx
        self.mlow = cmd.mlow
        self.mhigh = cmd.mhigh
        self.deltax = cmd.deltax
        self.EulerAngles = cmd.EulerAngles

class AllTildeB():
    def __init__(self, m, t):
        self.lmax = m
        self.tilde_b = t