#!/usr/bin/env python
# hetero.py
# Yunhan Wang (yw559)
# Oct 23, 2013
""" Further work:
1. finish the v00.5 codes and pass the instructions. It would be wonderful if everyone is clear what is stored in each variable and go through the test case step by step. Though this is time consuming, it would really benefit the further work and tests.
2. move to v00.7. The hetero.py is finished for this version.
3. isolate the python codes from Matlab. Currently the data required in part1, part2, and part3 are generated from Matlab, but the right way should be python part1 -> data -> python part2 -> data -> python part3. In detail:
the save and load should be redesigned and introduced into our codes.
Thus, some codes should be modified accordingly, especially when the data contains indexes. (we are using Matlab data, so the indexes taken in are subtracted by 1)
4. more test cases, or really careful writing of codes
5. optimization: refactor codes to improve the calculation efficiency, such as jroots_c

TEST:
    check each output file to be correct
    proof reading codes
Yunhan Wang 1/2014


Description: Same as hetero.m, parse all commands and parameters from instructions (inst_xx.py); for each command, call the corresponding
functions in funcs.py
Updated by Yunhan Wang 8/28/2014

"""
import numpy as np
from scipy import io
import funcs as fun
import helper as h
from structures import *
from numpy.fft import fft2, ifftshift
from scipy import misc



import numpy as np
import math
import helper as h
#import hetero as t
import sys
import scipy.linalg
# import ty_nnz
# import ty_any
# import ty_isnumeric
import inspect
#import time as tm
#from EMAN2 import EMNumPy
 #from scipy import io
import scipy.special as s
from mpmath import besselj, bessely, legendre
# from structures import *
from numpy import linalg as LA
from math import pi, sqrt, sin, cos, acos, atan2, factorial
#from EMAN2 import EMData
#from mpi4py import MPI
#import pp
import scipy.io as sio

from PIL import Image




log = Log() # serves as diary() or save(), initially only print to console

def instructionsFHV(fn):
    cmd = MatList(fn)
    hetero(cmd)

def hetero(cmd):
    """ proceed the commands inside cmd list """
    # var declaration
    deltachi=[0,0] #cmd[i].samplingintervals
    Na='' #cmd[i].NaNb[0];
    Nb='' #cmd[i].NaNb[1];
    Ny4minimalset,iiminimalset,vkminimalset,ixminimalset,iyminimalset = [0,0,0,0,0]
    vkmag=np.arange(1).reshape((1,1)) #sqrt( vkminimalset(:,1).^2 + vkminimalset(:,2).^2 );
    vobj = np.ndarray(shape=(2,2), dtype=float) #fun.virusobj_read(cmd[i].fn_clnp,cmd[i].fn_nu,cmd[i].fn_q)
    all_tilde_b = [] # a list of type AllTildeB
    Ny4minimalset, iiminimalset, vkminimalset, ixminimalset, iyminimalset, vkmag = 0,0,0,0,0,0
    normalizemask, npixels = 0, 0
    annulussamplemean,annulussamplevariance = 0.0, 0.0
    Imagestack = np.ndarray(shape=(2,2), dtype=float)
    imagestack, imageindex = 0, 0
    meanofy = 0.0
    Rabc = np.ndarray(shape=(2,2), dtype=float)
    pixelnoisevar0 = 0.0
    pixelnoisevar,loglikelihood = 0, 0
    EM_MC, EM_iter = EMMC(), EMIter()
    real_space_cubes = [] # should be instance of RealSpaceCubes()
    magk,fsc, imagestack2, vobj2 = 0, 0, 0, 0
    rhobar, rxx = 0,0
    vk = None
    for i in range(len(cmd)):
        """
        if Imagestack.shape != (2,2):
            print "shape"
            print imagestack[0].shape
        """
        """
        print "Image Shape"
        print Imagestack.shape
        """
        if (not isinstance(cmd[i], io.matlab.mio5_params.mat_struct)) or len(cmd[i]._fieldnames) == 0:
            h.printf(log, 'hetero: ii %d no-op command'%(i+1)) # matlab starts from 1 while python from 0
            continue

        if cmd[i].operator==u'misc_diary':# 1c 1b 2b 3b
            h.printf(log, 'hetero: ii %d misc_diary fn_diary %s'%(i+1,cmd[i].fn_diary))
            if len(cmd[i].fn_diary) != 0:
                fun.diary(cmd[i].fn_diary)
            continue

        if cmd[i].operator==u'misc_addtomatlabpath':
            h.printf(log, 'hetero: ii %d misc_addtomatlabpath dn %s'%(i+1,cmd[i].dn))
            continue

        if cmd[i].operator==u'misc_setpseudorandomnumberseed' : # 1c 1b 2b
            h.printf(log, 'hetero: ii %d misc_setpseudorandomnumberseed pseudorandomnumberseed %s'
                   %(i+1,cmd[i].pseudorandomnumberseed)),
            if cmd[i].pseudorandomnumberseed<0:
                h.errPrint('hetero: cmd{%d}.pseudorandomnumberseed %d < 0'%(i+1,cmd[i].pseudorandomnumberseed))
            fun.rng(cmd[i].pseudorandomnumberseed) # ???
            continue

        if cmd[i].operator==u'misc_savepseudorandomnumberstate2file' :
            h.printf(log, 'hetero: ii %d misc_savepseudorandomnumberstate2file fn_pseudorandomnumberstate %s'
                   %(i+1,cmd[i].fn_pseudorandomnumberstate))
            continue

        if cmd[i].operator==u'misc_restorepseudorandomnumberstatefromfile' :
            h.printf(log, 'hetero: ii %d misc_restorepseudorandomnumberstatefromfile fn_pseudorandomnumberstate %s'
                   %(i+1,cmd[i].fn_pseudorandomnumberstate))
            continue

        if cmd[i].operator==u'misc_clearvariable' : # new in v7
            # requires cmd{ii}.variablename
            h.printf(log,'hetero: ii %d misc_clearvariable variablename %s'%(i+1,cmd[i].variablename))
            # clear(cmd{ii}.variablename)

        if cmd[i].operator==u'misc_changedirectory' :
            h.printf(log, 'hetero: ii %d misc_changedirectory dn %s'%(i+1,cmd[i].dn))
            working_dir = os.path.dirname(os.path.realpath(__file__))
            dn = os.path.join(working_dir,cmd[i].dn)
            if not os.path.exists(dn):
                os.makedirs(dn)
            os.chdir(dn)
            continue

        if cmd[i].operator==u'misc_return' :
            h.printf(log, 'hetero: ii %d misc_return'%(i+1))
            continue

        if cmd[i].operator==u'misc_keyboard' :
            h.printf(log, 'hetero: ii %d misc_keyboard'%(i+1))
            continue

        if cmd[i].operator==u'misc_save_workspace' : # 1c !!!! 1b 2b 3b
            # requires cmd{ii}.fn_workspace
            h.printf(log, 'hetero: ii %d misc_save_workspace fn_workspace %s'%(i+1,cmd[i].fn_workspace)),
            #state4rng=rng # make sure that the state of the pseudorandom number generator is in the workspace
            #fun.save(cmd[i].fn_workspace)
            continue

        if cmd[i].operator==u'misc_load_workspace' : # modified in v7
            h.printf(log, 'hetero: ii %d misc_load_workspace fn_workspace %s existingoverreload %d'%(i+1,cmd[i].fn_workspace, cmd[i].existingoverreload))
            """
            ATTENTION:
            Here must be some error about the random seed generation!
            """
            if cmd[i].existingoverreload == "True": # boolean or String?
                # state4rng=rng # make sure that the state of the pseudorandom number generator is in the workspace
                state4rng = 383511
                #fun.save(tmp_workspace)
                # load(cmd{ii}.fn_workspace);
                # load tmp_workspace;
                # delete tmp_workspace;
                fun.rng(state4rng)
            else:
                # load(cmd{ii}.fn_workspace)
                # if exist('state4rng','var')
                fun.rng(state4rng)
                h.printf(log,'hetero: misc_load_workspace: state4rng does not exist')
            continue

        if cmd[i].operator==u'misc_write_mrc' : # 3b
            h.printf(log, 'hetero: ii %d misc_write_mrc fn_write_mrc %s what2write %s'%(i+1,cmd[i].fn_write_mrc,cmd[i].what2write))
            if cmd[i].what2write == 'write_image_stack':
                if np.amax(deltachi) != np.amin(deltachi):
                    h.errPrint('hetero: misc_write_mrc: deltachi %g %g'%(deltachi[0],deltachi[1]))

                m = np.zeros((imagestack[0][0].shape[0],imagestack[0][0].shape[1],len(imagestack)))

                for jj in range(len(imagestack)):
                    m[:,:,jj]=imagestack[jj][0]

                    
                fun.WriteMRC(imagestack,cmd[i].fn_write_mrc)

                   

            elif cmd[i].what2write == 'write_rhobar':
                if np.amax(real_space_cubes.deltax) != np.amin(real_space_cubes.deltax):
                    h.errPrint('hetero: misc_write_mrc: deltax %g %g %g'
                               %(real_space_cubes.deltax[0],real_space_cubes.deltax[1],real_space_cubes.deltax[2]))
                fun.WriteMRC(rhobar,cmd[i].fn_write_mrc)
            elif cmd[i].what2write == 'write_rxx':
                if np.amax(real_space_cubes.deltax) != np.amin(real_space_cubes.deltax):
                    h.errPrint('hetero: misc_write_mrc: deltax %g %g %g'
                               %(real_space_cubes.deltax[0],real_space_cubes.deltax[1],real_space_cubes.deltax[2]))
                fun.WriteMRC(rxx,cmd[i].fn_write_mrc)
            else:
                h.errPrint('hetero: ii %d misc_write_mrc: what2write %s'%(i+1,cmd[i].what2write))
            continue

        if cmd[i].operator==u'write_Image_stack': # new in v7
            continue

        if cmd[i].operator==u'misc_push' : # 3b
            h.printf(log, 'hetero: ii %d misc_push what2push %s'%(i+1,cmd[i].what2push))
            if cmd[i].what2push == 'push_image_stack':
                imagestack2=imagestack
            elif cmd[i].what2push == 'push_virusobj':
                vobj2=vobj
            else:
                h.errPrint('hetero: ii %d misc_push: what2push %s'%(i+1, cmd[i].what2push))
            continue

        if cmd[i].operator==u'misc_pop' : # 3b
            h.printf(log, 'hetero: ii %d misc_pop what2pop %s'%(i+1,cmd[i].what2pop))
            if cmd[i].what2pop == 'pop_image_stack':
                imagestack=imagestack2
            elif cmd[i].what2pop == 'pop_virusobj':
                vobj=vobj2
            else:
                h.errPrint('hetero: ii %d misc_pop: what2pop %s'%(i+1, cmd[i].what2pop))
            continue

        if cmd[i].operator=='misc_pack': # new in v7
            h.printf(log,'hetero: ii %d misc_pack'%(i+1))
            continue

        ##### basic operators:
        if cmd[i].operator==u'basic_set2Drealspacesamplingintervals' : # 1c 1b
            h.printf(log, 'hetero: ii %d basic_set2Drealspacesamplingintervals deltachi(1) %g deltachi(2) %g'
                   %(i+1,cmd[i].samplingintervals[0],cmd[i].samplingintervals[1])),
            deltachi=np.asarray(cmd[i].samplingintervals)
            continue

        if cmd[i].operator==u'basic_setsizeof2Drealspaceimages' : # 1c 1b
            h.printf(log, 'hetero: ii %d basic_setsizeof2Drealspaceimages Na %d Nb %d'
                   %(i+1,cmd[i].NaNb[0],cmd[i].NaNb[1]))
            Na=cmd[i].NaNb[0];
            Nb=cmd[i].NaNb[1];
            continue

        if cmd[i].operator==u'basic_setsizeof2Drealspaceimagesfromimagestack' : # 2b
            h.printf(log, 'hetero: ii %d basic_setsizeof2Drealspaceimagesfromimagestack'%(i+1))
            (Na, Nb)=imagestack[0][0].shape # imagestack[0] is vertical vector!
            continue

        if cmd[i].operator==u'basic_setsizeof2DreciprocalspaceImagesfromImagestack': # new in v7
            h.printf(log,'hetero: ii %d basic_setsizeof2DreciprocalspaceImagesfromImagestack'%(i+1))

            Na=Imagestack[0][0].shape[0]
            Nb=Imagestack[0][0].shape[1]
            continue

        if cmd[i].operator==u'basic_compute2Dreciprocalspaceproperties' : # 1c 1b 2b
            h.printf(log, 'hetero: ii %d basic_compute2Dreciprocalspaceproperties'%(i+1))
            # Determine a minimal subset of 2-D reciprocal space such that
            # conjugate symmetry fills in all of reciprocal space.  Assume
            # that all images have the same dimensions and extract the
            # dimensions from the first of the boxed 2-D real-space images.
            recipobj = fun.vk_indexset(Na,Nb,deltachi[0],deltachi[1])
            Ny4minimalset = recipobj[0]
            iiminimalset = recipobj[1]
            vkminimalset = recipobj[2]
            ixminimalset = recipobj[3]
            iyminimalset = recipobj[4]
            vkmag = recipobj[5]
            continue

        ##### real-space boxed image operators:
        if cmd[i].operator==u'box_readpixelnoisevar': # new in v7
            h.printf(log,'hetero: ii %d box_readpixelnoisevar fn_pixelnoisevar %s'%(i+1,cmd[i].fn_pixelnoisevar))
            try:
                with open(cmd[i].fn_pixelnoisevar,'r') as fid:
                    for line in fid:
                        pixelnoisevar = float(line)
            except IOError:
                h.errPrint('hetero: ii %d box_readpixelnoisevar fn_pixelnoisevar %s\n'%(i+1,cmd[i].fn_pixelnoisevar))

            h.printf(log,'hetero: ii %d box_readpixelnoisevar pixelnoisevar %g\n'%(i+1,pixelnoisevar))
            continue

        if cmd[i].operator==u'box_writepixelnoisevar' : # changed in v7 2b
            h.printf(log,'hetero: ii %d box_writepixelnoisevar fn_pixelnoisevar %s'%(i+1,cmd[i].fn_pixelnoisevar))
            try:
                with open(cmd[i].fn_pixelnoisevar, 'w') as fid:
                    fid.write('%g\n'%(pixelnoisevar))
                    # error('hetero: ii %d box_writepixelnoisevar status %d ~= 0\n',ii,status)
                    # clear fid status
            except IOError:
                h.errPrint('hetero: ii %d box_writepixelnoisevar: fn_pixelnoisevar %s'%(i+1,cmd[i].fn_pixelnoisevar))

        if cmd[i].operator==u'box_printimagesasarrays' :
            h.printf(log, 'hetero: ii %d box_printimagesasarrays: imageindex:'%(i+1)) # not sufficient
            continue

        if cmd[i].operator==u'box_readImagestack' :
            h.printf(log, 'hetero: ii %d box_readimagestack: imagestackformat %s fn_imagestack %s'%(i+1,cmd[i].Imagestackformat, cmd[i].fn_Imagestack))
            if cmd[i].Imagestackformat == u'fake':
                """
                #TODO I do not really implement readreadread here
                imagestack=readreadread(cmd{ii}.fn_imagestack);
                imageindex=[1:length(imagestack)]';
                """
                pass
            elif cmd[i].Imagestackformat == u'mat':
                # requires also cmd{ii}.startSlice
                # requires also cmd{ii}.numSlices %Number of real images.
                # Need two real images for each complex reciprocal-space image.

		#print 'hetero: ii %d box_readimagestack: mrc: startSlice %d numSlices %d'%(i+1,cmd[i].startSlice,cmd[i].numSlices)

                # this part fixed by Yiming Jia, March 6th, 2015

                #e = EMData()

                #e = EMData(cmd[i].fn_imagestack,0)

                #map_data = EMNumPy.em2numpy(e)

                #map_data = map_data.transpose()

                  #if map_data.shape[2]!=cmd[i].numSlices*2:
                    #sys.exit('hetero: ii %d box_readimagestack: mrc: size(map,3) %d\n'%(i+1,map_data.shape[2]))

                #Imagestack=np.zeros((map_data.shape[2]/2,1),dtype=np.object)

                #for j in range(Imagestack.size):
                    #Imagestack[j][0]=map_data[:,:,j*2]+1j*map_data[:,:,2*j+1]

                #Imageindex=np.array(range(cmd[i].startSlice-1,cmd[i].startSlice+cmd[i].numSlices-2))

                data = sio.loadmat(cmd[i].fn_Imagestack)
                Imagestack = data['Imagestack']
                if 'imageindex' in data:
                    imageindex = data['imageindex']



            elif cmd[i].Imagestackformat == u'img':
                """ATTENTION
                I did not implement the reading image part
                """
                pass
            else:
                sys.exit('hetero: ii %d unknown imagestackformat %s\n'%(i,cmd[i].Imagestackformat))
            continue

        if cmd[i].operator==u'box_readimagestack' :
            h.printf(log, 'hetero: ii %d box_readimagestack: imagestackformat %s fn_imagestack %s'%(i+1,cmd[i].imagestackformat, cmd[i].fn_imagestack))
            if cmd[i].imagestackformat == u'fake':
                """
                #TODO I do not really implement readreadread here
                imagestack=readreadread(cmd{ii}.fn_imagestack);
                imageindex=[1:length(imagestack)]';
                """
                pass
            elif cmd[i].imagestackformat == u'mat':
                # requires also cmd{ii}.startSlice
                # requires also cmd{ii}.numSlices
                #print 'hetero: ii %d box_readimagestack: mrc: startSlice %d numSlices %d'%(i+1,cmd[i].startSlice,cmd[i].numSlices)

                #(map_data,_,_,_,_) = fun.ReadMRC(str(cmd[i].fn_imagestack),cmd[i].startSlice,cmd[i].numSlices)


                #if map_data.shape[2]!=cmd[i].numSlices:
                    #sys.exit('hetero: ii %d box_readimagestack: mrc: size(map,3) %d\n'%(i+1,map_data.shape[2]))

                #imagestack=np.zeros((map_data.shape[2],1),dtype=np.object)
                #for j in range(imagestack.size):
                    #imagestack[j][0]=map_data[:,:,j]  # assignment failed, by Yunhan Wang 8/15/2014
                    #print map_data[:,:,j], map_data.shape ##### Yunhan
                #imageindex=np.array(range(cmd[i].startSlice-1,cmd[i].startSlice+cmd[i].numSlices-2))


                data = sio.loadmat(cmd[i].fn_imagestack)
                imagestack = data['imagestack']
                if 'imageindex' in data:
                    imageindex = data['imageindex']


            elif cmd[i].imagestackformat == u'img':
                """ATTENTION
                I did not implement the reading image part
                """
                pass
            else:
                sys.exit('hetero: ii %d unknown imagestackformat %s\n'%(i,cmd[i].imagestackformat))
            continue

        if cmd[i].operator==u'box_loadimagestack' : # 2b
            h.printf(log, 'hetero: ii %d box_loadimagestack: fn_imagestack %s'%(i+1,cmd[i].fn_imagestack))
            (l1, l2) = fun.load(cmd[i].fn_imagestack,'imagestack','imageindex')
            imagestack = np.asarray(l1)
            imageindex = np.asarray([i-1 for i in l2]) # index starting from 0
            continue

        if cmd[i].operator==u'box_loadImagestack': # new in v7
            h.printf(log,'hetero: ii %d box_loadImagestack: fn_Imagestack %s'%(i+1,cmd[i].fn_Imagestack))
            continue

        if cmd[i].operator==u'box_saveimagestack' : # 1c 1b
            # requires cmd{ii}.fn_imagestack
            h.printf(log, 'hetero: ii %d box_saveimagestack: fn_imagestack %s'%(i+1,cmd[i].fn_imagestack))

            dictimg = {}
            dictimg['imagestack'] = imagestack
            if imageindex:
                dictimg['imageindex'] = imageindex

            sio.savemat(cmd[i].fn_imagestack,dictimg)

            #fun.save(cmd[i].fn_imagestack,'imagestack','imageindex')  ?????
            continue

        if cmd[i].operator==u'box_saveImagestack': # new in v7
            h.printf(log,'hetero: ii %d box_saveImagestack: fn_Imagestack %s'%(i+1,cmd[i].fn_Imagestack))

            dictImg = {}
            dictImg['Imagestack'] = Imagestack
            if imageindex:
                dictImg['imageindex'] = imageindex

            sio.savemat(cmd[i].fn_Imagestack,dictImg)

            continue

        if cmd[i].operator==u'box_extractsubset' :
            h.printf(log, 'hetero: ii %d box_extractsubset: maxNv %d a %d b %d\n'%(i+1,cmd[i].maxNv,cmd[i].a,cmd[i].b))
            continue

        if cmd[i].operator==u'box_permute' :
            h.printf(log, 'hetero: ii %d box_permute'%(i+1))
            continue

        if cmd[i].operator==u'box_shrink' :
            h.printf(log, 'hetero: ii %d box_shrink: pixels2delete %d %d %d %d'
                   %(i+1,cmd[i].pixels2delete[0],cmd[i].pixels2delete[1],cmd[i].pixels2delete[2],cmd[i].pixels2delete[3]))
            continue

        if cmd[i].operator==u'box_maskcorners' :
            h.printf(log, 'hetero: ii %d box_maskcorners: radiuscorner %g'%(i+1,cmd[i].radiuscorner))
            continue

        if cmd[i].operator==u'box_normalize2zeroone' :
            h.printf(log, 'hetero: ii %d box_normalize2zeroone: radius01 %g %g'%(i+1,cmd[i].radius01[0],cmd[i].radius01[1]))
            continue

        if cmd[i].operator==u'box_classifyviasamplemean' :
            h.printf(log, 'hetero: ii %d box_classifyviasamplemean: classifythres %g'%(i+1,cmd[i].classifythres))
            continue

        if cmd[i].operator==u'box_annulusstatistics' : # 2b
            # requires cmd{ii}.radius01(1:2)
            # must be done collectively over all images.
            # all images must be the same size.
            h.printf(log, 'hetero: ii %d box_annulusstatistics: radius01 %g %g'%(i+1,cmd[i].radius01[0],cmd[i].radius01[1]))
            normalizemask=fun.box_normalize_mask(imagestack[0][0].shape,deltachi,cmd[i].radius01)
            npixels=len(fun.find_operators(normalizemask,True,"="))
            h.printf(log,'hetero: ii %d set2Drealspacesamplingintervals: number of pixels in the annulus %d'%(i+1,npixels))
            annulussamplemean=0.0
            for jj in range(len(imagestack)):
                annulussamplemean+=np.sum(imagestack[jj][0][normalizemask])

            annulussamplemean=annulussamplemean/(npixels*len(imagestack))
            annulussamplevariance=0.0
            for jj in range(len(imagestack)):
                annulussamplevariance+=np.sum(pow((imagestack[jj][0][normalizemask]-annulussamplemean), 2))
            annulussamplevariance=annulussamplevariance/(npixels*len(imagestack)-1)
            h.printf(log,'hetero: ii %d box_annulusstatistics: annulussamplemean %g annulussamplevariance %g'%(i+1,annulussamplemean,annulussamplevariance))
            continue

        if cmd[i].operator==u'realrecip_2DFFT' : # 2b
            h.printf(log, 'hetero: ii %d basic_realrecip_2DFFT'%(i+1))

            Imagestack = np.ndarray((len(imagestack),1), dtype=np.object) # each cell in Imagestack is a ndarray
            for ii in range(Imagestack.shape[0]):
                Imagestack[ii][0]=deltachi[0]* deltachi[1]* np.fft.fft2(np.fft.ifftshift(imagestack[ii][0]))
            continue

        if cmd[i].operator==u'box_writeannulusstatistics': # new in v7
            h.printf(log,'hetero: ii %d box_writeannulusstatistics fn_annulusstatistics %s'%(i+1,cmd[i].fn_annulusstatistics))
            try:
                with open(cmd[i].fn_annulusstatistics,'w') as fid:
                    s = "%g %g\n"%(annulussamplemean,annulussamplevariance)
                    fid.write(s)
                # error('hetero: ii %d post_write_FSC status %d != 0\n'%(i+1,status))
            except IOError:
                h.errPrint('hetero: ii %d box_writeannulusstatistics: fn_annulusstatistics %s\n'%(i+1,cmd[i].fn_annulusstatistics))
            continue

        if cmd[i].operator==u'box_readannulusstatistics': # new in v7
            h.printf(log,'hetero: ii %d box_readannulusstatistics fn_annulusstatistics %s'%(i+1,cmd[i].fn_annulusstatistics))
            continue

        ##### reciprocal-space boxed image operators:
        if cmd[i].operator==u'Box_printImagesasarrays' :
            h.printf(log, 'hetero: ii %d Box_printImagesasarrays: imageindex:'%(i+1))
            continue

        ##### virus object operators:
        if cmd[i].operator==u'vobj_print_virusobj' : # 2b
            # TODO
            h.printf(log, 'hetero: ii %d vobj_print_virusobj'%(i+1))
            for eta in range(len(vobj)):
                h.printf(log,'vobj_print_virusobj: eta %d clnp_fn %s nu_fn %s q_fn %s'%(eta+1,vobj[eta].clnp_fn,vobj[eta].nu_fn,vobj[eta].q_fn))
                h.printf(log,'vobj_print_virusobj: eta %d clnp.il:'%(eta+1))
                h.disp(log, vobj[eta].clnp.l.T)
                h.printf(log,'vobj_print_virusobj: eta %d clnp.in:'%(eta+1))
                h.disp(log, vobj[eta].clnp.n.T)
                h.printf(log,'vobj_print_virusobj: eta %d clnp.ip:'%(eta+1))
                h.disp(log, vobj[eta].clnp.p.T)
                h.printf(log,'vobj_print_virusobj: eta %d clnp.optflag:'%(eta+1))
                h.disp(log, vobj[eta].clnp.optflag.T)
                h.printf(log,'vobj_print_virusobj: eta %d clnp.c:'%(eta+1))
                h.disp(log, vobj[eta].clnp.c.T)
                h.printf(log,'vobj_print_virusobj: eta %d cbar:'%(eta+1))
                h.disp(log, vobj[eta].cbar.T)
                h.printf(log,'vobj_print_virusobj: eta %d BasisFunctionType %s'%(eta+1,vobj[eta].BasisFunctionType)) ### Peter
                h.printf(log,'vobj_print_virusobj: eta %d R1 %g R2 %g'%(eta+1,vobj[eta].R1,vobj[eta].R2))
                h.printf(log,'vobj_print_virusobj: eta %d nu:'%(eta+1))
                h.disp(log, vobj[eta].nu.T)
                h.printf(log,'vobj_print_virusobj: eta %d q %g'%(eta+1,vobj[eta].q))
            # TODO display
            continue

        if cmd[i].operator==u'vobj_read_virusobj' : # 1c 1b 2b
            # requires cmd{ii}.fn_clnp, cell array of Neta file names
            # requires cmd{ii}.fn_nu, cell array of Neta file names
            # requires cmd{ii}.fn_q, cell array of Neta file names

            if not isinstance(cmd[i].fn_clnp, (frozenset, list, set, tuple,)):
                h.printf(log, 'hetero: ii %d vobj_read_virusobj fn_clnp %s fn_nu %s fn_q %s'%(i+1,cmd[i].fn_clnp,cmd[i].fn_nu,cmd[i].fn_q))
                print len(cmd[i].fn_clnp)
            else:
                for jj in range(1, len(cmd[i].fn_clnp)):
                    h.printf(log, 'hetero: ii %d vobj_read_virusobj fn_clnp %s fn_nu %s fn_q %s'%(i+1,cmd[i].fn_clnp[jj],cmd[i].fn_nu[jj],cmd[i].fn_q[jj]))

            vobj=fun.virusobj_read(cmd[i].fn_clnp,cmd[i].fn_nu,cmd[i].fn_q)
            # clear jj

            continue

        if cmd[i].operator==u'vobj_write_virusobj' :

           h.printf(log, 'hetero: ii %d vobj_write_virusobj'%(i+1))

           for eta in range(len(vobj)):
               try:
                    with open(cmd[i].fn_clnp[eta], 'w') as fid:
                        s1 = "%d\n"%(vobj[eta].BasisFunctionType[0])
                        s2 = "%g %g\n"%(vobj[eta].R1,vobj[eta].R2)
                        fid.write(s1)
                        fid.write(s2)
                        for idx in range(len(vobj[eta].clnp.l)):
                            c = "%d %d %d %d %g\n" % (vobj[eta].clnp.l[idx], vobj[eta].clnp.n[idx], vobj[eta].clnp.p[idx], vobj[eta].clnp.optflag[idx],vobj[eta].clnp.c[idx])
                            fid.write(c)

               except IOError:
                    h.errPrint('virusobj_write: eta %d fopen fn_clnp %s\n'%(eta,cmd[i].fn_clnp[eta]))


               try:
                    with open(cmd[i].fn_nu[eta], 'w') as fid:
                        if(vobj[eta].nu.size != 0):
                            for idx in range(len(vobj[eta].nu)):
                                 s = "%g\n"%(vobj[eta].nu[idx])
                                 fid.write(s)
                        else:
                            print("dont know hot to deal with unix(['touch ' fn_nu{eta}]);")
               except IOError:
                    h.errPrint('virusobj_write: eta %d fopen fn_nu %s\n'%(eta,cmd[i].fn_nu[eta]))

               try:
                    with open(cmd[i].fn_q[eta], 'w') as fid:
                            s = "%g\n"%(vobj[eta].q)
                            fid.write(s)
               except IOError:
                    h.errPrint('virusobj_write: eta %d fopen fn_q %s\n' % (eta, cmd[i].fn_q[eta]))
           continue



        if cmd[i].operator==u'vobj_save_virusobj' : # 2b ???? no save
            h.printf(log, 'hetero: ii %d vobj_save_virusobj fn_virusobj %s'%(i+1,cmd[i].fn_virusobj))
            #save(cmd[i].fn_virusobj,'vobj')
            continue

        if cmd[i].operator==u'vobj_load_virusobj' : # 3b
            h.printf(log, 'hetero: ii %d vobj_load_virusobj fn_virusobj %s'%(i+1,cmd[i].fn_virusobj))
            vobj = fun.load(cmd[i].fn_virusobj,'vobj')[0]
            continue

        if cmd[i].operator==u'vobj_change_size_of_virusobj' : # 2b
            # requires cmd{ii}.vlmax(1:Neta)
            # requires cmd{ii}.vpmax(1:Neta)
            # Change the size of lmax and pmax in the virus model that will be used.  Does not set 2Dreciprocal.
            h.printf(log, 'hetero: ii %d vobj_change_size_of_virusobj:'%(i+1)),
            h.printf(log, 'hetero: ii %d vlmax:'%(i+1))
            h.disp(log, cmd[i].vlmax)
            h.printf(log, 'hetero: ii %d vpmax:'%(i+1))
            h.disp(log, cmd[i].vpmax)
            vobj=fun.virusobj_changesize(cmd[i].vlmax,cmd[i].vpmax,vobj)
            continue

        if cmd[i].operator==u'vobj_change_homo2hetero_in_virusobj' : # 2b
            h.printf(log, 'hetero: ii %d vobj_change_homo2hetero_in_virusobj:'%(i+1)),
            h.printf(log, 'hetero: ii %d homo2hetero:'%(i+1))
            for eta in range(cmd[i].homo2hetero.size):
                h.printf(log,'hetero: ii %d vobj_change_homo2hetero_in_virusobj eta %d action %d'%(i+1,eta,cmd[i].homo2hetero[eta].action))
            vobj=fun.virusobj_change_homo2hetero(vobj,cmd[i].homo2hetero)
            continue
        
        # supplemented by Guantian
        if cmd[i].operator==u'vobj_change_handedness':
            # requires cmd{ii}.changehand(1:Neta)
            for eta in range(len(vobj)):
                if cmd[i].changehand[eta]:
                    for tochange in range(len(vobj[eta].clnp.l)):
                        if vobj[eta].clnp.l[tochange] % 2 == 1:
                            vobj[eta].clnp.c[tochange] *= -1
                    vobj[eta].cbar = vobj[eta].clnp.c
            print_vobj(vobj, i)
            continue
        
       # supplemented by Guantian
        if cmd[i].operator=='vobj_change_sign':
            # requires cmd{ii}.changesign(1:Neta)
            for eta in range(len(vobj)):
                if cmd[i].changesign[eta]:
                    vobj[eta].clnp.c *= -1
                    vobj[eta].cbar = vobj[eta].clnp.c
            print_vobj(vobj, i)
            continue

        ##### integration rules:
        if cmd[i].operator==u'quad_read_integration_rule' : # 1c 1b 2b
            # requires cmd{ii}.fn_rule file name
            h.printf(log, 'hetero: ii %d quad_read_integration_rule fn_rule %s'%(i+1,cmd[i].fn_rule))
            rule=fun.rd_rule(cmd[i].fn_rule)
            continue

        ##### forward operators:
        if cmd[i].operator==u'fw_mk_synthetic_2D_realspace_images' : # 1c
            h.printf(log, 'hetero: ii %d fw_mk_synthetic_2D_realspace_images Nv %d NT %d SNR %g'
                   %(i+1,cmd[i].Nv,cmd[i].NT,cmd[i].SNR))
            (y,imagestack,truevalues,pixelnoisevar)=fun.fw(cmd[i].SNR,vobj,vkminimalset,cmd[i].Nv,cmd[i].NT,Na,Nb,rule,all_tilde_b,ixminimalset,iyminimalset) # y has no noise
            #np.savetxt(truevalues,truevalues,fmt="%d %d")


            sio.savemat('imagestack.mat',{'imagestack':imagestack})
            #sio.savemat('inst_Neta2_rule49_Nv500_fw.mat',{'cmd':cmd})

            print(imagestack[3][0])
            a= imagestack[3][0];
            image = Image.new("1", (91, 91))
            pixels = image.load()
            for i in range(image.size[0]):
                for j in range(image.size[1]):
                    pixels[i, j] = a[i][j]

            image.show()

            # imageindex=[1:length(imagestack)]'
            #print (y[1,0])

            continue

        if cmd[i].operator==u'fw_save_truevalues' : # 1c 1b
            # requires cmd{ii}.fn_truevalues
            h.printf(log, 'hetero: ii %d fw_save_truevalues fn_truevalues %s'%(i+1,cmd[i].fn_truevalues))

            #fun.save(cmd[i].fn_truevalues,'truevalues') ??????
            continue

        if cmd[i].operator==u'fw_write_truevalues':
            # requires cmd{ii}.fn_truevalues
            h.printf(log, 'hetero: ii %d fw_write_truevalues fn_truevalues %s'%(i+1,cmd[i].fn_truevalues))
            np.savetxt(cmd[i].fn_truevalues,truevalues,fmt="%d %d")

            continue

        ##### expectation-maximization operators:
        if cmd[i].operator==u'EM_read_tilde_b' : # 1c 1b 2b 3b
            # requires cmd{ii}.fn_tilde_b file name
            h.printf(log, 'hetero: ii %d EM_read_tilde_b fn_tilde_b %s'%(i+1,cmd[i].fn_tilde_b))
            for eta in range(len(vobj)):
                m = np.amax(vobj[eta].clnp.l)
                print cmd[i].fn_tilde_b
                t = fun.rd_b(cmd[i].fn_tilde_b,m)
                all_tilde_b.append(AllTildeB(m,t))

            #all_tilde_b
            # clear eta
            continue

        if cmd[i].operator==u'EM_extractdatasubset' : # changed in v7 2b
            # Construct the reciprocal space image data structure for the range of reciprocal space $\|\vec\kappa\|<kmax$ that will be used.
            h.printf(log, 'hetero: ii %d EM_extractdatasubset kmax %g'%(i+1,cmd[i].kmax))
            [vk,y]=fun.subset_reciprocalspace(cmd[i].kmax,vkmag,vkminimalset,Imagestack,iiminimalset)
            h.printf(log,'hetero: ii %d EM_extractdatasubset Ny=size(vk,1)=%s'%(i+1,vk.shape[0]))
            continue

        if cmd[i].operator==u'EM_set_2Dreciprocal_in_virusobj' : # 1c 1b 2b
            # requires cmd{ii}.use_vkminimalset_rather_than_vk
            h.printf(log, 'hetero: ii %d EM_set_2Dreciprocal_in_virusobj use_vkminimalset_rather_than_vk %d'
                   %(i+1,cmd[i].use_vkminimalset_rather_than_vk))
            if cmd[i].use_vkminimalset_rather_than_vk:
                vobj=fun.virusobj_set_2Dreciprocal(vobj,vkminimalset)
            else:
                vobj=fun.virusobj_set_2Dreciprocal(vobj,vk)
            continue

        if cmd[i].operator==u'EM_rm_2Dreciprocal_in_virusobj' :
            h.printf(log, 'hetero: ii %d EM_rm_2Dreciprocal_in_virusobj'%(i+1))
            continue

        if cmd[i].operator==u'EM_sphericalsymmetry_homogeneous' : # 2b
            # least squares
            h.printf(log, 'hetero: ii %d EM_sphericalsymmetry_homogeneous'%(i+1))
            # A spherically-symmetric object has a pure-real Fourier transform.  Therefore,
            # such a model can make only a 0 prediction of the imaginary components of the data.
            # Imaginary components of the data are removed from y and the corresponding rows of L are removed.
            meanofy=np.sum(y,axis=1)/y.shape[1] # compute meanofy before deleting imaginary components.
            meanofy=meanofy[0::2]
            Rabc=np.eye(3)
            for eta in range(len(vobj)):
                L=fun.setL_nord(Rabc, vobj[eta].clnp.l, vobj[eta].clnp.n, vk, vobj[eta].Htable, vobj[eta].map_unique2lp, all_tilde_b[eta].tilde_b)
                L=L[0::2,:]
                vobj[eta].cbar = np.dot(np.linalg.pinv(np.dot(L.T,L)), np.dot(L.T,meanofy))
                vobj[eta].clnp.c = vobj[eta].cbar
            continue

        if cmd[i].operator==u'EM_expectationmaximization' : # changed in v7 2b
            # can do homogeneous or heterogeneous cases, can do various symmetries or no symmetry
            h.printf(log, 'hetero: ii %d EM_expectationmaximization'%(i+1))

            # Package the parameters related to the Monte Carlo choice of initial condition in a structure for simplicity
            EM_MC.Nrandic=cmd[i].MC_Nrandic
            EM_MC.FractionOfMeanForMinimum=cmd[i].MC_FractionOfMeanForMinimum
            EM_MC.FractionOfMean=cmd[i].MC_FractionOfMean

            # Package the parameters related to the Expectation-Maximization iterations in a structure for simplicity.
            EM_iter.maxiter=cmd[i].maxiter
            EM_iter.maxiter4pixelnoisevarupdate=cmd[i].maxiter4pixelnoisevarupdate
            EM_iter.cbarftol=cmd[i].cbarftol
            EM_iter.cbarrtol=cmd[i].cbarrtol
            EM_iter.cbarftol_dividebyNc=cmd[i].cbarftol_dividebyNc
            EM_iter.loglikeftol=cmd[i].loglikeftol
            EM_iter.loglikertol=cmd[i].loglikertol
            EM_iter.nu_ic_FractionOfcbarForMinimum=cmd[i].nu_ic_FractionOfcbarForMinimum
            EM_iter.nu_ic_FractionOfcbar=cmd[i].nu_ic_FractionOfcbar
            EM_iter.estimate_noise_var_in_homogeneous_problem=cmd[i].estimate_noise_var_in_homogeneous_problem
            EM_iter.nu_ic_always_proportional2cbar=cmd[i].nu_ic_always_proportional2cbar
            EM_iter.rule=rule
            EM_iter.Na=Na
            EM_iter.V_TolX=cmd[i].V_TolX
            EM_iter.V_MaxIter=cmd[i].V_MaxIter
            EM_iter.fn_savehistory=cmd[i].fn_savehistory
            EM_iter.verbosity=cmd[i].verbosity
            EM_iter.MinimumClassProb=cmd[i].MinimumClassProb
            s = cmd[i].pixelnoisevar_initialcondition
            if s == 'from_image_statistics':
                # For the following formula, please see test_noisevar.m.  The
                # fact that the reciprocal space image is complex and the
                # code treats the Re and Im parts as separate measurements
                # (independent and with equal variance) leads to the factor
                # of 0.5.  The user must have already set Na and Nb by using
                # one of 'basic_setsizeof2Drealspaceimages',
                # 'basic_setsizeof2Drealspaceimagesfromimagestack', or
                # 'basic_setsizeof2DreciprocalspaceImagesfromImagestack'.
                pixelnoisevar0=0.5*Na*Nb*annulussamplevariance
                h.printf(log,'hetero: ii %d EM_expectationmaximization: pixelnoisevar0 %g annulussamplevariance %g Na %d Nb %d'%(i+1,pixelnoisevar0,annulussamplevariance,Na,Nb))
            elif s == 'from_pixelnoisevar':
                pixelnoisevar0=pixelnoisevar
                h.printf(log,'hetero: ii %d EM_expectationmaximization: pixelnoisevar0 %g pixelnoisevar %g'%(i+1,pixelnoisevar0,pixelnoisevar))
            else:
                h.errPrint('hetero: ii %d EM_expectationmaximization: unknown pixelnoisevar_initialcondition %s'%(i+1,cmd[i].pixelnoisevar_initialcondition))
            h.printf(log,'hetero: ii %d EM_expectationmaximization: cmd{ii}.MultiplierForPixelnoisevarIC %g\n'%(i+1,cmd[i].MultiplierForPixelnoisevarIC))
            pixelnoisevar0=pixelnoisevar0*cmd[i].MultiplierForPixelnoisevarIC
            """The for statement below are to adjust the data structure difference"""
            for eta in range(len(vobj)):
                if len(vobj[eta].cbar.shape) != 2:
                    vobj[eta].cbar = vobj[eta].cbar.reshape(-1,1)
                if len(vobj[eta].clnp.c.shape) != 2:
                    vobj[eta].clnp.c = vobj[eta].clnp.c.reshape(-1,1)
                if len(vobj[eta].clnp.l.shape) != 2:
                    vobj[eta].clnp.l = vobj[eta].clnp.l.reshape(-1,1)
                if len(vobj[eta].clnp.n.shape) != 2:
                    vobj[eta].clnp.n = vobj[eta].clnp.n.reshape(-1,1)
                if len(vobj[eta].clnp.p.shape) != 2:
                    vobj[eta].clnp.p = vobj[eta].clnp.p.reshape(-1,1)
                if len(vobj[eta].nu) != 2:
                    vobj[eta].nu = vobj[eta].nu.reshape(-1,1)
            vobj,pixelnoisevar,loglikelihood=fun.EM_expmax_MonteCarlo(vk,y,vobj,pixelnoisevar0,EM_MC,EM_iter,all_tilde_b)
            h.printf(log,'hetero: ii %d EM_expectationmaximization: pixelnoisevar %g'%(i+1,pixelnoisevar))
            continue

        ##### post-processing operators:set_Ttable_nord
        if cmd[i].operator==u'post_compute_real_space_cubes' : # 3b
            # requires cmd{ii}.whichclass
            # requires cmd{ii}.wantrhobar
            # requires cmd{ii}.wantrxx
            # requires cmd{ii}.mlow(1:3)
            # requires cmd{ii}.mhigh(1:3)
            # requires cmd{ii}.deltax(1:3)
            # requires cmd{ii}.EulerAngles(1:3)
            h.printf(log, 'hetero: ii %d post_compute_real_space_cubes'%(i+1))

            wc = cmd[i].whichclass - 1 # here index is subtracted by 1

            if cmd[i].whichclass < 1 or len(vobj) < cmd[i].whichclass:
                h.errPrint('hetero: ii %d post_compute_real_space_cubes: whichclass %d'%(i+1,cmd[i].whichclass))
            [rhobar,rxx,x_rect] = fun.plt_realspace(cmd[i].wantrhobar,cmd[i].wantrxx,vobj[wc], all_tilde_b[wc],
                                        cmd[i].mlow[0],cmd[i].mhigh[0],cmd[i].deltax[0],
                                        cmd[i].mlow[1],cmd[i].mhigh[1],cmd[i].deltax[1],
                                        cmd[i].mlow[2],cmd[i].mhigh[2],cmd[i].deltax[2],
                                        cmd[i].EulerAngles[0],cmd[i].EulerAngles[1],cmd[i].EulerAngles[2])
            # Package the parameters for simplicity
            real_space_cubes = RealSpaceCubes(cmd[i]) # whichclass is the index!
            continue

        if cmd[i].operator==u'post_save_real_space_cubes' : # 3b
            h.printf(log, 'hetero: ii %d post_save_real_space_cubes fn_real_space_cubes %s'
                   %(i+1,cmd[i].fn_real_space_cubes))
            # save(cmd[i].fn_real_space_cubes,'rhobar','rxx','x_rect','real_space_cubes')
            continue

        if cmd[i].operator==u'post_compute_FSC' : # 3b
            h.printf(log, 'hetero: ii %d post_compute_FSC minmagk %g maxmagk %g deltamagk %g eta4classA %d eta4classB %d is_same_vobj %d'
                   %(i+1,cmd[i].FSC_minmagk,cmd[i].FSC_maxmagk,cmd[i].FSC_deltamagk, cmd[i].FSC_eta4classA,cmd[i].FSC_eta4classB,cmd[i].FSC_is_same_vobj))
            if cmd[i].FSC_is_same_vobj:
                # FSC_eta4classA B are indexes!
                [magk,fsc] = fun.get_FSC(vobj[cmd[i].FSC_eta4classA-1], vobj[cmd[i].FSC_eta4classB-1],
                           cmd[i].FSC_minmagk, cmd[i].FSC_maxmagk, cmd[i].FSC_deltamagk)
            else:
                h.printf(log, 'hetero: ii %d post_compute_FSC: class A in vobj2 and class B in vobj'%(i+1))
                [magk,fsc] = fun.get_FSC(vobj2[cmd[i].FSC_eta4classA-1], vobj[cmd[i].FSC_eta4classB-1],
                           cmd[i].FSC_minmagk, cmd[i].FSC_maxmagk, cmd[i].FSC_deltamagk)
            continue

        if cmd[i].operator==u'post_save_FSC' : # 3b
            h.printf(log, 'hetero: ii %d post_save_FSC fn_FSC %s'%(i+1,cmd[i].fn_FSC))
            # save(cmd[i].fn_FSC,'magk','fsc')
            continue

        if cmd[i].operator==u'post_write_FSC' : # 3b
            h.printf(log, 'hetero: ii %d post_write_FSC fn_FSC %s'%(i+1,cmd[i].fn_FSC))
            try:
                with open(cmd[i].fn_FSC,'w') as fid:
                    for ii in range(fsc.shape[0]):
                        s = str(magk[ii])+" "+" ".join(map(str,fsc[ii]))+"\n"
                        fid.write(s)
                # error('hetero: ii %d post_write_FSC status %d != 0\n'%(i+1,status))
            except IOError:
                h.errPrint('hetero: ii %d post_write_FSC fid error'%(i+1))
            continue

        # default
        h.printf(log, 'hetero: ii %d unknown operator %s'%(i+1, cmd[i].operator))

if __name__=='__main__':
    """ do not run this file. Use inst_XXXX.py instead
    updated by Yunhan Wang, 9/8/2014
    """
    #instructionsFHV("fhv1b.mat")
    #instructionsFHV("fhv2b.mat")
    instructionsFHV("fhv3b.mat")
