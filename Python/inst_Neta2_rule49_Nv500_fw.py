#!/usr/bin/env python

import scipy.io as sio
import numpy as np
import hetero
"""
matlab Copyright 2013 Yili Zheng, Qiu Wang, Peter C. Doerschuk
python Copyright 2014 Shenghan Gao, Yayi Li, Yu Tang, Peter C. Doerschuk
Cornell University has not yet decided on the license for this software so no rights come with this file.
Certainly no warrenty of any kind comes with this file.
All of this will be corrected when Cornell University comes to a decision.
"""

def inst_Neta2_rule49_Nv500_fw():
    outputbasename='FHV.out.lmax10pmax5.Neta2.rule49.Nv500.fw'
    print 'inst_Neta2_rule49_Nv500_fw: outputbasename ' + outputbasename
    """
    Operators are executed in the order in which they appear in cmd, using the arguments that also appear in cmd.
    To do nothing, specify cmd=[];
    To indicate a 'no-op', set that element of the cell array to an empty matrix, e.g., cmd{1}=[], which is the initialization set by the 'cell' function.
    """
    cmd=[]

    ####################
    cmd.append({'operator':'misc_diary', 'fn_diary':outputbasename+'.diary'})
    ####################
    cmd.append({'operator':'misc_setpseudorandomnumberseed', 'pseudorandomnumberseed':383511})
    ####################
    cmd.append({'operator':'basic_set2Drealspacesamplingintervals', 'samplingintervals':[4.7,4.7]})
    ####################
    cmd.append({'operator':'basic_setsizeof2Drealspaceimages', 'NaNb':[91 ,91]})
    ####################
    cmd.append({'operator':'basic_compute2Dreciprocalspaceproperties'})
    ####################
    cmd.append({'operator':'quad_read_integration_rule', 'fn_rule':'rule_small_3_rulechopper'})
    ####################
    cmd_dict = {}
    eta=2;
    cmd_dict['operator']='vobj_read_virusobj'
    cmd_dict['fn_clnp']=np.zeros((2,1), dtype=np.object)
    cmd_dict['fn_clnp'][0]='FHV.lmax10pmax5.clnp.txt'
    cmd_dict['fn_clnp'][1]='FHV.lmax10pmax5.clnp.perturbed.txt'
    cmd_dict['fn_nu']=np.zeros((2,1), dtype=np.object)
    cmd_dict['fn_nu'][0]='FHV.lmax10pmax5.nu.txt'
    cmd_dict['fn_nu'][1]='FHV.lmax10pmax5.nu.perturbed.txt'
    cmd_dict['fn_q']=np.zeros((2,1), dtype=np.object)
    cmd_dict['fn_q'][0]='FHV.lmax10pmax5.q.txt'
    cmd_dict['fn_q'][1]='FHV.lmax10pmax5.q.perturbed.txt'
    cmd.append(cmd_dict)
    ####################
    cmd.append({'operator':'EM_set_2Dreciprocal_in_virusobj', 'use_vkminimalset_rather_than_vk':True})
    ####################
    cmd.append({'operator':'EM_read_tilde_b', 'fn_tilde_b':'callti.out.read_by_C'})
    ####################

    cmd_dict = {}
    cmd_dict['operator']='fw_mk_synthetic_2D_realspace_images'
    cmd_dict['Nv']=500
    cmd_dict['NT']=1
    cmd_dict['SNR']=5.0
    cmd.append(cmd_dict)
    ####################
    cmd.append({'operator':'fw_write_truevalues', 'fn_truevalues':outputbasename+'.truevalues.txt'})
    ####################
    cmd.append({'operator':'box_saveimagestack', 'fn_imagestack':outputbasename+'.imagestack.py.mat'})
    ####################
    cmd.append({'operator':'misc_diary', 'fn_diary':'off'})
    ####################
    #Execute the operations in the cmd cell array.

    sio.savemat('inst_Neta2_rule49_Nv500_fw.mat',{'cmd':cmd})

if __name__ == "__main__":
    inst_Neta2_rule49_Nv500_fw()
    cmd = sio.loadmat('inst_Neta2_rule49_Nv500_fw.mat',squeeze_me=True, struct_as_record=False)['cmd']

    hetero.hetero(cmd)