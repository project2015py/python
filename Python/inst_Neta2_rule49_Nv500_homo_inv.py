#!/usr/bin/env python

import scipy.io as sio
import numpy as np
import hetero
from funcs import set_kbiggest
import math
"""
matlab Copyright 2013 Yili Zheng, Qiu Wang, Peter C. Doerschuk
python Copyright 2014 Shenghan Gao, Yayi Li, Yu Tang, Peter C. Doerschuk
Cornell University has not yet decided on the license for this software so no rights come with this file.
Certainly no warrenty of any kind comes with this file.
All of this will be corrected when Cornell University comes to a decision.
"""

def inst_Neta2_rule49_Nv500_homo_inv():
    #####################
    outputbasename='FHV.out.lmax10pmax5.Neta2.rule49.Nv500.homo.inv'
    print 'inst_Neta2_rule49_Nv500_homo_inv: outputbasename ' + outputbasename
    #####################
    Neta=2
    deltachi=[4.7, 4.7] # image sampling intervals in Angstroms
    R2=197.4 # outer radius R2
    """
    Operators are executed in the order in which they appear in cmd,
    using the arguments that also appear in cmd.  To do nothing, specify cmd=[];.
    To indicate a 'no-op', set that element of the cell array to an empty matrix,
    .g., cmd{1}=[], which is the initialization set by the 'cell' function.
    """
    #####################
    cmd=[] #Preallocate for 100 operators.
    #####################
    cmd.append({'operator':'misc_diary', 'fn_diary':outputbasename+'.diary'})
    #####################
    cmd.append({'operator':'misc_setpseudorandomnumberseed', 'pseudorandomnumberseed':29831})
    #####################
    cmd_dict = {}
    Neta=2;
    cmd_dict['operator']='box_readimagestack'
    cmd_dict['imagestackformat']='mat'
    cmd_dict['fn_imagestack']='FHV.out.lmax10pmax5.Neta2.rule49.Nv500.fw.imagestack.py.mat'
    cmd.append(cmd_dict)
    #####################NOT IN pre4hetero_inv
    cmd.append({'operator':'basic_setsizeof2Drealspaceimagesfromimagestack'})
    #####################
    cmd.append({'operator':'basic_set2Drealspacesamplingintervals', 'samplingintervals':deltachi})
    #####################
    cmd.append({'operator':'box_annulusstatistics', 'radius01':[R2+deltachi[0], R2+10*deltachi[0]]})
    #####################NOT IN pre4hetero_inv
    cmd.append({'operator':'basic_compute2Dreciprocalspaceproperties'})
    #####################
    #ii=8
    cmd.append({'operator':'realrecip_2DFFT'})
    #####################
    cmd_dict = {}
    cmd_dict['operator']='vobj_read_virusobj'
    cmd_dict['fn_clnp']=np.zeros((2,1), dtype=np.object)
    cmd_dict['fn_clnp'][0]='FHV.ic.lmax0pmax1.clnp.c001iszero.txt'
    cmd_dict['fn_clnp'][1]='FHV.ic.lmax0pmax1.clnp.perturbed.c001iszero.txt'
    cmd_dict['fn_nu']=np.zeros((2,1), dtype=np.object)
    cmd_dict['fn_nu'][0]='FHV.ic.lmax0pmax1.nu.homogeneous.txt'
    cmd_dict['fn_nu'][1]='FHV.ic.lmax0pmax1.nu.homogeneous.txt'
    cmd_dict['fn_q']=np.zeros((2,1), dtype=np.object)
    cmd_dict['fn_q'][0]='FHV.ic.lmax0pmax1.q.equalclassprobs.txt'
    cmd_dict['fn_q'][1]='FHV.ic.lmax0pmax1.q.equalclassprobs.txt'
    cmd.append(cmd_dict)
    #####################
    cmd.append({'operator':'vobj_print_virusobj'})
    ##################################################
    #Increase the number of coefficients.
    ##################################################
    #####################
    cmd_dict = {}
    cmd_dict['operator']='vobj_change_size_of_virusobj'
    cmd_dict['vlmax']=[10,10] # vector of new lmax values, one for each class
    cmd_dict['vpmax']=[5,5] # vector of new pmax values, one for each class
    cmd.append(cmd_dict)
    #####################
    cmd.append({'operator':'vobj_print_virusobj'})
    #####################
    cmd.append({'operator':'EM_read_tilde_b', 'fn_tilde_b':'callti.out.read_by_C'})
    #####################
    cmd_dict = {}
    cmd_dict['operator']='vobj_change_size_of_virusobj'
    cmd_dict['vlmax']=[0,0] # vector of new lmax values, one for each class
    cmd_dict['vpmax']=[4,4] # vector of new pmax values, one for each class
    cmd.append(cmd_dict)
    #####################
    cmd.append({'operator':'EM_extractdatasubset', 'kmax':-1})
    #####################
    cmd.append({'operator':'EM_set_2Dreciprocal_in_virusobj','use_vkminimalset_rather_than_vk':False})
    #####################
    cmd.append({'operator':'misc_diary', 'fn_diary':'off'})
    ##################################################
    #####################
    #*******************Solve the linear least squares problem for a one-class spherically-symmetric homogeneous-problem.
    cmd.append({'operator':'misc_changedirectory', 'dn':'step0'})
    #####################
    cmd.append({'operator':'misc_diary', 'fn_diary':outputbasename+'.diary.txt'})
    #####################
    cmd.append({'operator':'EM_sphericalsymmetry_homogeneous'})
    #####################
    cmd.append({'operator':'vobj_print_virusobj'})
    #####################
    # cmd.append({'operator':'vobj_save_virusobj','fn_virusobj':outputbasename+'.vobj.mat'})
    #####################
    cmd_dict = {}
    cmd_dict['operator']='vobj_write_virusobj'
    cmd_dict['fn_clnp']=np.zeros((2,1), dtype=np.object)
    cmd_dict['fn_clnp'][0]=outputbasename+'.eta1.clnp.txt'
    cmd_dict['fn_clnp'][1]=outputbasename+'.eta2.clnp.txt'
    cmd_dict['fn_nu']=np.zeros((2,1), dtype=np.object)
    cmd_dict['fn_nu'][0]=outputbasename+'.eta1.nu.txt'
    cmd_dict['fn_nu'][1]=outputbasename+'.eta2.nu.txt'
    cmd_dict['fn_q']=np.zeros((2,1), dtype=np.object)
    cmd_dict['fn_q'][0]=outputbasename+'eta1.q.txt'
    cmd_dict['fn_q'][1]=outputbasename+'eta2.q.txt'
    cmd.append(cmd_dict)
    #####################
    cmd.append({'operator':'misc_diary', 'fn_diary':'off'})
    #####################
    cmd.append({'operator':'misc_changedirectory', 'dn':'..'})
    #*******************Done solving the linear least squares problem
    #####################
    cmd.append({'operator':'misc_diary', 'fn_diary':outputbasename+'.diary2.txt'})
    #####################
    """
    Define the resolution steps used in YinZhengDoerschukNatarajanJohnson JSB 2003 Table 1.
    In the sh/C code used in YinZhengDoerschukNatarajanJohnson JSB 2003, the spatial frequency vector with largest magnitude was computed by directly finding the vector in vkminimalset with maximum magnitude.  vkminimalset was computedd by vk_indexset.m which contains the function index2k.
    """
    ZhyeYinresolutionsteps={'lmax':np.array([10,15,21,25,31,36,45]),
                            'pmax':np.array([6,8,8,10,10,15,20]),
                            'Nic':np.array([100,10,10,1,1,1,1]),
                            'kmaxpow':np.array([3,3,3,2,1,1,1]),
                            'kmax':np.zeros((1,7))}
    deltachi=[4.7,4.7]; # YinZhengDoerschukNatarajanJohnson JSB 2003 FHV example
    NaNb=[91,91] # YinZhengDoerschukNatarajanJohnson JSB 2003 FHV example
    kbiggest=math.sqrt(set_kbiggest(NaNb[0],deltachi[0])**2 + set_kbiggest(NaNb[1],deltachi[1])**2)
    ZhyeYinresolutionsteps['kmax']=kbiggest*np.exp((-ZhyeYinresolutionsteps['kmaxpow']/2.)*math.log(2.))
    #####################
    """
    Scale back the size of the computing for this synthetic problem.
    Do the lmax=0 case because this is a 2 class problem while the initial linear estimator is 1 class.
    """
    ZhyeYinresolutionsteps['lmax']=np.array([0,6,10])
    ZhyeYinresolutionsteps['pmax']=np.array([4,5,5])
    # ZhyeYinresolutionsteps['Nic']=np.array([100,100,10]) # This is for inv.m
    ZhyeYinresolutionsteps['Nic']=np.array([1,10,10]) # This is for inv.RUN1.m
    ZhyeYinresolutionsteps['kmaxpow']=ZhyeYinresolutionsteps['kmaxpow'][0:3]
    ZhyeYinresolutionsteps['kmax']=ZhyeYinresolutionsteps['kmax'][0:3]
    #####################
    # MultiplierForPixelnoisevarIC_eachstep=np.array([10.0,1.0,1.0]) # This is for inv.m
    MultiplierForPixelnoisevarIC_eachstep=np.array([10.0,1.0,1.0])
    #####################
    cmd.append({'operator':'misc_diary','fn_diary':'off'})
    #####################









    #*******************Begin computing homogeneous reconstructions at increasing resolutions.
    #for step in range(ZhyeYinresolutionsteps['lmax'].size):
    for step in range(ZhyeYinresolutionsteps['lmax'].size):
        #####################
        cmd.append({'operator':'misc_changedirectory', 'dn':'step'+str(step+1)})
        #####################
        cmd.append({'operator':'misc_diary','fn_diary':outputbasename+'.diary.txt'})
        #####################
        cmd_dict = {}
        cmd_dict['operator']='vobj_change_size_of_virusobj'
        cmd_dict['vlmax']=[ZhyeYinresolutionsteps['lmax'][step], ZhyeYinresolutionsteps['lmax'][step]] # vector of new lmax values, one for each class
        cmd_dict['vpmax']=[ZhyeYinresolutionsteps['pmax'][step], ZhyeYinresolutionsteps['pmax'][step]] #vector of new pmax values, one for each class
        cmd.append(cmd_dict)
        #####################  
        cmd.append({'operator':'vobj_print_virusobj'}) 
        #####################
         #cmd.append({'operator':'EM_read_tilde_b','fn_tilde_b':'callti.out.read_by_C'})
        #cmd.append({'operator':'EM_read_tilde_b', 'fn_tilde_b':'callti.out.read_by_C'})
        #####################
        cmd.append({'operator':'EM_extractdatasubset','kmax':ZhyeYinresolutionsteps['kmax'][step]})
        #####################
        cmd.append({'operator':'EM_set_2Dreciprocal_in_virusobj','use_vkminimalset_rather_than_vk':False})
        #####################
        cmd.append({'operator':'quad_read_integration_rule','fn_rule':'/Users/mengyuan/Downloads/Python/rule_small_3_rulechopper'})
        #####################
        cmd_dict = {}
        cmd_dict['operator']='EM_expectationmaximization'
        cmd_dict['MC_Nrandic']=ZhyeYinresolutionsteps['Nic'][step]
        cmd_dict['MC_FractionOfMeanForMinimum']=0.005
        cmd_dict['MC_FractionOfMean']=0.2
        cmd_dict['maxiter']=200
        cmd_dict['maxiter4pixelnoisevarupdate']=0
        cmd_dict['cbarftol']=-1.0 # unused.
        cmd_dict['cbarrtol']=1.0e-4
        cmd_dict['cbarftol_dividebyNc']=False
        cmd_dict['loglikeftol']=-1.0 # unused.
        cmd_dict['loglikertol']=-1.0 # unused.
        cmd_dict['nu_ic_FractionOfcbarForMinimum']=0.005
        cmd_dict['nu_ic_FractionOfcbar']=0.1
        cmd_dict['estimate_noise_var_in_homogeneous_problem']=False
        cmd_dict['pixelnoisevar_initialcondition']='from_image_statistics'
        cmd_dict['nu_ic_always_proportional2cbar']=[]
        cmd_dict['V_TolX']=np.NaN
        cmd_dict['V_MaxIter']=np.NaN
        cmd_dict['fn_savehistory']=outputbasename+'.history.mat'
        cmd_dict['verbosity']=1
        cmd_dict['MultiplierForPixelnoisevarIC']=MultiplierForPixelnoisevarIC_eachstep[step]
        cmd_dict['MinimumClassProb']=1.0e-3/Neta
        cmd.append(cmd_dict)
        #####################
        cmd.append({'operator':'box_writepixelnoisevar','fn_pixelnoisevar':outputbasename+'.pixelnoisevar.txt'})
        #####################
        cmd.append({'operator':'vobj_print_virusobj'})
        #####################
        # cmd.append({'operator':'vobj_save_virusobj','fn_virusobj':outputbasename+'.vobj.mat'})
        #####################
        cmd_dict = {}
        cmd_dict['operator']='vobj_write_virusobj'
        cmd_dict['fn_clnp']=np.zeros((2,1), dtype=np.object)
        cmd_dict['fn_clnp'][0]=outputbasename+'.eta1.clnp.txt'
        cmd_dict['fn_clnp'][1]=outputbasename+'.eta2.clnp.txt'
        cmd_dict['fn_nu']=np.zeros((2,1), dtype=np.object)
        cmd_dict['fn_nu'][0]=outputbasename+'.eta1.nu.txt'
        cmd_dict['fn_nu'][1]=outputbasename+'.eta2.nu.txt'
        cmd_dict['fn_q']=np.zeros((2,1), dtype=np.object)
        cmd_dict['fn_q'][0]=outputbasename+'.eta1.q.txt'
        cmd_dict['fn_q'][1]=outputbasename+'.eta2.q.txt'
        cmd.append(cmd_dict)
        #####################
        cmd.append({'operator':'misc_diary','fn_diary':'off'})
        #####################
        cmd.append({'operator':'misc_changedirectory','dn':'..'})
        #####################
    #*******************End computing homogeneous reconstructions at increasing resolutions.
    #####################


    sio.savemat('inst_Neta2_rule49_Nv500_homo_inv.mat',{'cmd':cmd},long_field_names=True)


if __name__ == "__main__":
    inst_Neta2_rule49_Nv500_homo_inv()
    cmd = sio.loadmat('inst_Neta2_rule49_Nv500_homo_inv.mat',squeeze_me=True, struct_as_record=False)['cmd']

    hetero.hetero(cmd)