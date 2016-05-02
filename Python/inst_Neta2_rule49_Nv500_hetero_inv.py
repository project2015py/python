#!/usr/bin/env python

import scipy.io as sio
import numpy as np
import hetero
from funcs import set_kbiggest
import os
"""
matlab Copyright 2013 Yili Zheng, Qiu Wang, Peter C. Doerschuk
python Copyright 2014 Shenghan Gao, Yayi Li, Yu Tang, Peter C. Doerschuk
Cornell University has not yet decided on the license for this software so no rights come with this file.
Certainly no warrenty of any kind comes with this file.
All of this will be corrected when Cornell University comes to a decision.
"""
"""
Loads reciprocal space image stack.
Because there is only one kmax in this script,
it is possible to prepare y in an earlier script and not have to store both Imagestack and y.
"""
def inst_Neta2_rule49_Nv500_hetero_inv():
    #####################
    inputbasename='FHV.out.lmax10pmax5.Neta2.rule49.Nv500.homo.inv'
    outputbasename='FHV.out.lmax10pmax5.Neta2.rule49.Nv500.hetero.inv'
    print 'inst_Neta2_rule49_Nv500_hetero_inv: outputbasename '+outputbasename
    #####################
    NaNb=[91,91] # image dimensions in pixels
    deltachi=[4.7,4.7] # image sampling intervals in Angstroms
    Neta=2
    #####################
    # Operators are executed in the order in which they appear in cmd, using the arguments that also appear in cmd.  To do nothing, specify cmd=[];.  To indicate a 'no-op', set that element of the cell array to an empty matrix, e.g., cmd{1}=[], which is the initialization set by the 'cell' function.
    cmd=[] # Preallocate for 50 operators.
    #####################
    cmd.append({'operator':'misc_diary', 'fn_diary':outputbasename+'.diary'})
    #####################
    cmd.append({'operator':'box_readpixelnoisevar', 'fn_pixelnoisevar':'step3/'+inputbasename+'.pixelnoisevar.txt'})
    #####################
    cmd_dict={}
    cmd_dict['operator']='box_readImagestack'
    cmd_dict['Imagestackformat']='mat'
    cmd_dict['fn_Imagestack']='FHV.out.lmax10pmax5.Neta2.rule49.Nv500.pre4hetero.inv.Imagestack.mat'
    #cmd_dict['startSlice']=1
    #cmd_dict['numSlices']=500 # this is number of complex reciprocal-space images
    cmd.append(cmd_dict)
    #####################
    cmd.append({'operator':'basic_setsizeof2DreciprocalspaceImagesfromImagestack'})
    #####################
    cmd.append({'operator':'basic_set2Drealspacesamplingintervals','samplingintervals':deltachi})
    #####################
    cmd.append({'operator':'basic_compute2Dreciprocalspaceproperties'})
    #####################
    cmd_dict={}
    cmd_dict['operator']='vobj_read_virusobj'
    cmd_dict['fn_clnp']=np.zeros((Neta,1),dtype=np.object)
    cmd_dict['fn_clnp'][0]='step3/'+inputbasename+'.eta1.clnp.txt'
    cmd_dict['fn_clnp'][1]='step3/'+inputbasename+'.eta2.clnp.txt'
    cmd_dict['fn_nu']=np.zeros((Neta,1),dtype=np.object)
    cmd_dict['fn_nu'][0]='step3/'+inputbasename+'.eta1.nu.txt'
    cmd_dict['fn_nu'][1]='step3/'+inputbasename+'.eta2.nu.txt'
    cmd_dict['fn_q']=np.zeros((Neta,1),dtype=np.object)
    cmd_dict['fn_q'][0]='step3/'+inputbasename+'.eta1.q.txt'
    cmd_dict['fn_q'][1]='step3/'+inputbasename+'.eta2.q.txt'
    cmd.append(cmd_dict)
    #####################
    cmd_dict={}
    cmd_dict['operator']='vobj_change_homo2hetero_in_virusobj'
    cmd_dict['homo2hetero']=[]
    homo_dict={}
    cmd_dict['homo2hetero'].append(homo_dict)
    homo_dict['action']=1 # make this class heterogeneous
    homo_dict['FractionOfMeanForMinimum']=0.0025 # only required when .action=1
    homo_dict['FractionOfMean']=0.02 # only required when .action=1
    homo_dict={}
    cmd_dict['homo2hetero'].append(homo_dict)
    homo_dict['action']=1 # make this class heterogeneous
    homo_dict['FractionOfMeanForMinimum']=0.0025 # only required when .action=1
    homo_dict['FractionOfMean']=0.02 # only required when .action=1
    cmd.append(cmd_dict)
    #####################
    cmd.append({'operator':'vobj_print_virusobj'})
    #####################
    cmd.append({'operator':'misc_diary','fn_diary':'off'})
    #####################
    cmd.append({'operator':'misc_changedirectory','dn':'hetero.step7'})
    #####################
    cmd.append({'operator':'misc_diary','fn_diary':outputbasename+'.diary.txt'})
    #####################
    cmd.append({'operator':'EM_read_tilde_b','fn_tilde_b':'/Users/mengyuan/Downloads/Python/callti.out.read_by_C'})
    #####################
    cmd_dict={}
    cmd_dict['operator']='EM_extractdatasubset'
    kbiggest=(set_kbiggest(NaNb[0],deltachi[0])**2 + set_kbiggest(NaNb[1],deltachi[1])**2)**0.5
    cmd_dict['kmax']=kbiggest*0.5/(2**0.5)
    cmd.append(cmd_dict)
    #####################
    cmd.append({'operator':'EM_set_2Dreciprocal_in_virusobj','use_vkminimalset_rather_than_vk':False})
    #####################
    cmd.append({'operator':'quad_read_integration_rule','fn_rule':'/Users/mengyuan/Downloads/Python/rule_small_3_rulechopper'})
    #####################
    cmd_dict={}
    cmd_dict['operator']='EM_expectationmaximization'
    cmd_dict['MC_Nrandic']=1
    cmd_dict['MC_FractionOfMeanForMinimum']=0.005 # unused because MC_Nrandic=1 but still must be defined.
    cmd_dict['MC_FractionOfMean']=0.2 # unused because MC_Nrandic=1 but still must be defined.
    cmd_dict['maxiter']=200
    cmd_dict['maxiter4pixelnoisevarupdate']=0
    cmd_dict['cbarftol']=-1.0; # unused.
    cmd_dict['cbarrtol']=1.0e-4
    cmd_dict['cbarftol_dividebyNc']=False;
    cmd_dict['loglikeftol']=-1.0 # unused.
    cmd_dict['loglikertol']=-1.0 # unused.
    cmd_dict['nu_ic_FractionOfcbarForMinimum']=0.15
    cmd_dict['nu_ic_FractionOfcbar']=0.1
    cmd_dict['estimate_noise_var_in_homogeneous_problem']=False
    cmd_dict['pixelnoisevar_initialcondition']='from_pixelnoisevar'
    cmd_dict['nu_ic_always_proportional2cbar']=[]
    cmd_dict['V_TolX']=1e-10
    cmd_dict['V_MaxIter']=8
    cmd_dict['fn_savehistory']=outputbasename+'.history.mat'
    cmd_dict['verbosity']=1
    cmd_dict['MultiplierForPixelnoisevarIC']=1.0
    cmd_dict['MinimumClassProb']=10e-3/Neta
    cmd.append(cmd_dict)
    #####################
    cmd.append({'operator':'box_writepixelnoisevar','fn_pixelnoisevar':outputbasename+'.pixelnoisevar.txt'})
    #####################
    cmd.append({'operator':'vobj_print_virusobj'})
    #####################
    """
    %ii=ii+1;
    %cmd{ii}.operator='vobj_save_virusobj';
    %cmd{ii}.fn_virusobj=[outputbasename '.vobj.mat'];
    """
    #####################
    cmd_dict={}
    cmd_dict['operator']='vobj_write_virusobj'
    cmd_dict['fn_clnp']=np.array((Neta,1),dtype=np.object)
    cmd_dict['fn_clnp'][0]=outputbasename+'.eta1.clnp.txt'
    cmd_dict['fn_clnp'][1]=outputbasename+'.eta2.clnp.txt'
    cmd_dict['fn_nu']=np.array((Neta,1),dtype=np.object)
    cmd_dict['fn_nu'][0]=outputbasename+'.eta1.nu.txt'
    cmd_dict['fn_nu'][1]=outputbasename+'.eta2.nu.txt'
    cmd_dict['fn_q']=np.array((Neta,1),dtype=np.object)
    cmd_dict['fn_q'][0]=outputbasename+'.eta1.q.txt'
    cmd_dict['fn_q'][1]=outputbasename+'.eta2.q.txt'
    cmd.append(cmd_dict)
    #####################
    cmd.append({'operator':'misc_diary','fn_diary':'off'})
    #####################
    cmd.append({'operator':'misc_changedirectory','dn':'..'})
    #####################
    # ii=ii+1;
    # cmd{ii}.operator='misc_save_workspace';
    # cmd{ii}.fn_workspace=[outputbasename '.workspace.mat'];
    #####################
    # Execute the operations in the cmd cell array.
    sio.savemat('inst_Neta2_rule49_Nv500_hetero_inv.mat',{'cmd':cmd},long_field_names=True)

if __name__ == "__main__":
    inst_Neta2_rule49_Nv500_hetero_inv()
    cmd = sio.loadmat('inst_Neta2_rule49_Nv500_hetero_inv.mat',squeeze_me=True, struct_as_record=False)['cmd']

    hetero.hetero(cmd)