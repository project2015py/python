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

def inst_Neta2_rule49_Nv500_pre4hetero_inv():
    # Writes reciprocal space image stack in mrc format.

    #####################
    outputbasename='FHV.out.lmax10pmax5.Neta2.rule49.Nv500.pre4hetero.inv'
    print 'inst_Neta2_rule49_Nv500_pre4hetero_inv: outputbasename '+outputbasename
    #####################
    deltachi=[4.7,4.7] # image sampling intervals in Angstroms
    R2=197.4 # outer radius R2
    #####################
    # Operators are executed in the order in which they appear in cmd, using the arguments that also appear in cmd.  To do nothing, specify cmd=[];.  To indicate a 'no-op', set that element of the cell array to an empty matrix, e.g., cmd{1}=[], which is the initialization set by the 'cell' function.
    cmd=[] # Preallocate for 100 operators.
    #####################
    cmd.append({'operator':'misc_diary', 'fn_diary':outputbasename+'.diary'})
    #####################
    cmd.append({'operator':'misc_setpseudorandomnumberseed', 'pseudorandomnumberseed':29831})
    #####################
    cmd_dict = {}
    cmd_dict['operator']='box_readimagestack'
    cmd_dict['imagestackformat']='mat'
    cmd_dict['fn_imagestack']='FHV.out.lmax10pmax5.Neta2.rule49.Nv500.fw.imagestack.py.mat'
    cmd.append(cmd_dict)
    #####################
    """
    %NwV-JSB2013 ii=ii+1;
    %NwV-JSB2013 cmd{ii}.operator='box_classifyviasamplemean';
    %NwV-JSB2013 cmd{ii}.classifythres=0.16; %WangMatsuiDomitrovicZhengDoerschukJohnson JSB 2013, p. 197, column 2 line -3
    %NwV-JSB2013 %%%%%%%%%%%%%%%%%%%%
    %NwV-JSB2013 ii=ii+1;
    %NwV-JSB2013 cmd{ii}.operator='box_permute';
    %NwV-JSB2013 %%%%%%%%%%%%%%%%%%%%
    %NwV-JSB2013 ii=ii+1;
    %NwV-JSB2013 cmd{ii}.operator='box_extractsubset';
    %NwV-JSB2013 cmd{ii}.maxNv=1200; %WangMatsuiDomitrovicZhengDoerschukJohnson JSB 2013, p. 198, column 1 line 6
    %NwV-JSB2013 cmd{ii}.a=1; %WangMatsuiDomitrovicZhengDoerschukJohnson JSB 2013, p. 198, column 1 line 7
    %NwV-JSB2013 cmd{ii}.b=4; %WangMatsuiDomitrovicZhengDoerschukJohnson JSB 2013, p. 198, column 1 line 7
    """
    #####################
    cmd.append({'operator':'basic_set2Drealspacesamplingintervals', 'samplingintervals':deltachi})
    #####################
    """
    %NwV-JSB2013 %lines 134-171 of /home/qw32/hetero3d/newYiliCode/newcode/cacRuns/NwV_cap_preprocess/cacfw_nwv_dV_ico.m concern the subtraction of the mean and scaling by the standard deviation.  The statistics are computed in the region > 250 Angstrom from the center of the image.  Note that Qiu Wang gives radii in terms of pixels not Angstroms.
    %NwV-JSB2013 ii=ii+1;
    %NwV-JSB2013 cmd{ii}.operator='box_normalize2zeroone';
    %NwV-JSB2013 cmd{ii}.radius01=[250 1000]; %1000 Angstrom is outside of the image, even in the corners
    """
    #####################
    cmd.append({'operator':'box_annulusstatistics','radius01':[R2+deltachi[0],R2+10*deltachi[0]]}) # Uncertain of the correspondence with Qiu Wang's code
    #####################
    cmd.append({'operator':'realrecip_2DFFT'})
    #####################
    """
    %ii=ii+1;
    %cmd{ii}.operator='misc_save_workspace';
    %cmd{ii}.fn_workspace=[outputbasename '.workspace.mat'];
    """
    #####################
    cmd.append({'operator':'box_writeannulusstatistics','fn_annulusstatistics':outputbasename+'.annulusstatistics.txt'})
    #####################image stack has not been modified
    """
    ii=ii+1;
    cmd{ii}.operator='box_saveImagestack';
    cmd{ii}.fn_write_mrc=[outputbasename '.Imagestack.mrc'];
    cmd{ii}.what2write='write_image_stack';
    """
    #####################
    cmd_dict = {}
    cmd_dict['operator']='box_saveImagestack'
    cmd_dict['fn_Imagestack']=outputbasename+'.Imagestack.mat'
    #cmd_dict['what2write']='write_image_stack' # otherwise will skip the writeMRC routine, by Yunhan Wang 08/14/2014
    cmd.append(cmd_dict)
    #####################
    cmd.append({'operator':'misc_diary','fn_diary':'off'})
    #####################
    sio.savemat('inst_Neta2_rule49_Nv500_pre4hetero_inv.mat',{'cmd':cmd})
    
if __name__ == "__main__":
    inst_Neta2_rule49_Nv500_pre4hetero_inv()
    cmd = sio.loadmat('inst_Neta2_rule49_Nv500_pre4hetero_inv.mat',squeeze_me=True, struct_as_record=False)['cmd']

    hetero.hetero(cmd)
