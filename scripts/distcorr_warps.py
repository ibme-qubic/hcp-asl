import os
import subprocess as sp
import sys
import os.path as op 
import glob 
import tempfile 
from pathlib import Path
import multiprocessing as mp
import argparse

import regtricks as rt
import nibabel as nb
import scipy.ndimage
import numpy as np
from fsl.wrappers import fslmaths, bet
from fsl.data.image import Image
from scipy.ndimage import binary_fill_holes


def generate_asl2struct_initial(asl_vol0, struct, fsdir, reg_dir):
    """
    Generate the initial linear transformation between ASL-space and T1w-space
    using FS bbregister. This is further refined later on using asl_reg. Note
    that struct is required only for saving the output in the right convention, 
    it is not actually used by bbregister. 
    
    Args:
        asl_vol0: path to first volume of ASL 
        struct: path to T1w image (eg T1w_acdc_restore.nii.gz)
        fsdir: path to subject's FreeSurfer output directory 
        reg_dir: path to registration directory, for output 

    Returns: 
        n/a, file 'asl2struct_initial_bbr_fsl.mat' will be saved in reg_dir
    """

    # We need to do some hacky stuff to get bbregister to work...
    # Split the path to the FS directory into a fake $SUBJECTS_DIR
    # and subject_id. We temporarily set the environment variable
    # before making the call, and then revert back afterwards  
    new_sd, sid = op.split(fsdir)
    old_sd = os.environ.get('SUBJECTS_DIR')
    pwd = os.getcwd()
    orig_mgz = op.join(fsdir, 'mri', 'orig.mgz')

    # Run inside the regdir. Save the output in fsl format, by default 
    # this targets the orig.mgz, NOT THE T1 IMAGE ITSELF! 
    os.chdir(reg_dir)
    omat_path = op.join(reg_dir, "asl2orig_mgz_initial_bbr_fsl.mat")
    cmd = os.environ['SUBJECTS_DIR'] = new_sd
    cmd = f"$FREESURFER_HOME/bin/bbregister --s {sid} --mov {asl_vol0} --t1 "
    cmd += f"--reg asl2orig_mgz_initial_bbr.dat --fslmat {omat_path}"
    sp.run(cmd, shell=True)
    asl2orig_fsl = rt.Registration.from_flirt(omat_path, asl_vol0, orig_mgz)

    # Return to original working directory, and flip the FSL matrix to target
    # asl -> T1, not orig.mgz. Save output. 
    os.chdir(pwd)
    if old_sd:
        os.environ['SUBJECTS_DIR'] = old_sd
    asl2struct_fsl = asl2orig_fsl.to_flirt(asl_vol0, struct)
    np.savetxt(op.join(reg_dir, 'asl2struct_initial_bbr_fsl.mat'), asl2struct_fsl)

def generate_gdc_warp(asl_vol0, coeffs_path, distcorr_dir):
    """
    Generate distortion correction warp via gradient_unwarp. 

    Args: 
        asl_vol0: path to first volume of ASL series
        coeffs_path: path to coefficients file for the scanner (.grad)
        distcorr_dir: directory in which to put output
    
    Returns: 
        n/a, file 'fullWarp_abs.nii.gz' will be created in output dir
    """

    # Need to run in the output directory to make sure files end up in the
    # right place
    pwd = os.getcwd()
    os.chdir(distcorr_dir)
    cmd = ("gradient_unwarp.py {} gdc_corr_vol1.nii.gz siemens -g {}"
            .format(asl_vol0, coeffs_path))
    sp.run(cmd, shell=True)
    os.chdir(pwd)

def generate_wmmask(aparc_aseg):
    """ 
    Generate binary WM mask in space of T1 image using FS aparc+aseg

    Args: 
        aparc_aseg: path to aparc_aseg in T1 space (not FS 256 1mm space!)

    Returns: 
        np.array logical WM mask in space of T1 image 
    """

    aseg_array = nb.load(aparc_aseg).get_data()
    wm = np.logical_or(aseg_array == 41, aseg_array == 2)
    return wm 

def generate_topup_params(pars_filepath):
    """
    Generate a file containing the parameters used by topup
    """
    if os.path.isfile(pars_filepath):
        os.remove(pars_filepath)
    with open(pars_filepath, "a") as t_pars:
        t_pars.write("0 1 0 0.04845" + "\n")
        t_pars.write("0 -1 0 0.04845")

def generate_fmaps(pa_ap_sefms, params, config, distcorr_dir): 
    """
    Generate fieldmaps via topup for use with asl_reg. 

    Args: 
        asl_vol0: path to image of stacked blipped images (ie, PEdir as vol0,
            (oPEdir as vol1), in this case stacked as pa then ap)
        params: path to text file for topup --datain, PE directions/times
        config: path to text file for topup --config, other args 
        distcorr_dir: directory in which to put output
    
    Returns: 
        n/a, files 'fmap, fmapmag, fmapmagbrain.nii.gz' will be created in output dir
    """

    pwd = os.getcwd()
    os.chdir(distcorr_dir)

    # Run topup to get fmap in Hz 
    topup_fmap = op.join(distcorr_dir, 'topup_fmap_hz.nii.gz')        
    cmd = (("topup --imain={} --datain={}".format(pa_ap_sefms, params)
            + " --config={} --out=topup".format(config))
            + " --fout={} --iout={}".format(topup_fmap,
                op.join(distcorr_dir, 'corrected_sefms.nii.gz')))
    sp.run(cmd, shell=True)

    fmap, fmapmag, fmapmagbrain = [ 
        op.join(distcorr_dir, '{}.nii.gz'.format(s)) 
        for s in [ 'fmap', 'fmapmag', 'fmapmagbrain' ]
    ]    

    # Convert fmap from Hz to rad/s
    fmap_spc = rt.ImageSpace(topup_fmap)
    fmap_arr_hz = nb.load(topup_fmap).get_data()
    fmap_arr = fmap_arr_hz * 2 * np.pi
    fmap_spc.save_image(fmap_arr, fmap)

    # Mean across volumes of corrected sefms to get fmapmag
    fmapmag_arr = nb.load(op.join(
                    distcorr_dir, "corrected_sefms.nii.gz")).get_data()
    fmapmag_arr = fmapmag_arr.mean(-1)
    fmap_spc.save_image(fmapmag_arr, fmapmag)

    # Run BET on fmapmag to get brain only version 
    bet(fmap_spc.make_nifti(fmapmag_arr), output=fmapmagbrain)

    os.chdir(pwd)

def generate_asl_mask(struct_brain, asl, asl2struct):
    """
    Generate brain mask in ASL space 

    Args: 
        struct_brain: path to T1 brain-extracted, ac_dc_restore_brain
        asl: path to ASL image 
        asl2struct: regtricks.Registration for asl to structural 

    Returns: 
        np.array, logical mask. 
    """

    brain_mask = (nb.load(struct_brain).get_data() > 0).astype(np.float32)
    asl_mask = asl2struct.inverse().apply_to_array(brain_mask, struct_brain, asl)
    asl_mask = binary_fill_holes(asl_mask > 0.25)
    return asl_mask

def generate_epidc_warp(asl_vol0_brain, struct, struct_brain, asl_mask,
                       wmmask, asl2struct, fmap, fmapmag, fmapmagbrain, 
                       distcorr_dir):
    """
    Generate EPI distortion correction warp via asl_reg. 

    Args: 
        asl_vol0_brain: path to first volume of ASL series, brain-extracted
        struct: path to T1 image, ac_dc_restore
        struct_brain: path to brain-extracted T1 image, ac_dc_restore_brain
        asl_mask: path to brain mask in ASL space 
        wmmask: path to WM mask in T1 space 
        asl2struct: regtricks.Registration for asl to structural
        fmap: path to topup's field map in rad/s
        fmapmag: path to topup's field map magnitude 
        fmapmagbrain: path to topup's field map magnitude, brain only
        distcorr_dir: path to directory in which to place output 

    Returns: 
        n/a, file 'asl2struct_warp.nii.gz' is created in output directory 
    """

    a2s_fsl = op.join(distcorr_dir, 'asl2struct.mat')
    asl2struct.save_fsl(a2s_fsl, asl_vol0_brain, struct)
    cmd = ("asl_reg -i {} -o {} ".format(asl_vol0_brain, distcorr_dir)
           + "-s {} --sbet={} -m {} ".format(struct, struct_brain, asl_mask)
           + "--tissseg={} --imat={} --finalonly ".format(wmmask, a2s_fsl)
           + "--fmap={} --fmapmag={} ".format(fmap, fmapmag)
           + "--fmapmagbrain={} --pedir=y --echospacing=0.00057 ".format(fmapmagbrain))
    sp.run(cmd, shell=True)

def binarise_image(image, threshold=0):
    """
    Binarise image above a threshold if given.

    Args:
        image: path to the image to be binarised
        threshold: voxels with a value below this will be zero and above will be one
    
    Returns:
        np.array, logical mask
    """
    image = Image(image)
    mask = (image.data>0).astype(np.float32)
    return mask

def create_ti_image(asl, tis, sliceband, slicedt, outname):
    """
    Create a 4D series of actual TIs at each voxel.

    Args:
        asl: path to image in the space we wish to create the TI series
        tis: list of TIs in the acquisition
        sliceband: number of slices per band in the acquisition
        slicedt: time taken to acquire each slice
        outname: path to which the ti image is saved
    
    Returns:
        n/a, file outname is created in output directory
    """

    asl_spc = rt.ImageSpace(asl)
    n_slice = asl_spc.size[2]
    slice_in_band = np.tile(np.arange(0, sliceband), 
                            n_slice//sliceband).reshape(1, 1, n_slice, 1)
    ti_array = np.array([np.tile(x, asl_spc.size) for x in tis]).transpose(1, 2, 3, 0)
    ti_array = ti_array + (slice_in_band * slicedt)
    rt.ImageSpace.save_like(asl, ti_array, outname)

def main():

    # argument handling
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "study_dir",
        help="Path of the base study directory."
    )
    parser.add_argument(
        "sub_number",
        help="Subject number."
    )
    parser.add_argument(
        "-g",
        "--grads",
        help="Filename of the gradient coefficients for gradient"
            + "distortion correction (optional)."
    )
    parser.add_argument(
        "-t",
        "--target",
        help="Which space we want to register to. Can be either 'asl' for "
            + "registration to the first volume of the ASL series or "
            + "'structural' for registration to the T1w image. Default "
            + " is 'asl'.",
        default="asl"
    )
    parser.add_argument(
        "--fmap_ap",
        help="Filename for the AP fieldmap for use in distortion correction"
    )
    parser.add_argument(
        "--fmap_pa",
        help="Filename for the PA fieldmap for use in distortion correction"
    )

    args = parser.parse_args()
    study_dir = args.study_dir
    sub_id = args.sub_number
    grad_coefficients = args.grads
    target = args.target
    pa_sefm = args.fmap_pa
    ap_sefm = args.fmap_ap

    # For debug, re-use existing intermediate files 
    force_refresh = True

    # Input, output and intermediate directories
    # Create if they do not already exist. 
    sub_base = op.abspath(op.join(study_dir, sub_id))
    grad_coefficients = op.abspath(grad_coefficients)
    pvs_dir = op.join(sub_base, "T1w", "ASL", "PVEs")
    t1_asl_dir = op.join(sub_base, "T1w", "ASL")
    distcorr_dir = op.join(sub_base, "ASL", "TIs", "DistCorr")
    reg_dir = op.join(sub_base, 'T1w', 'ASL', 'reg')
    t1_dir = op.join(sub_base, "T1w")
    asl_dir = op.join(sub_base, "ASL", "TIs", "STCorr2")
    asl_out_dir = op.join(t1_asl_dir, "TIs", "DistCorr")
    calib_out_dir = op.join(t1_asl_dir, "Calib", "Calib0", "DistCorr")
    [ os.makedirs(d, exist_ok=True) 
        for d in [pvs_dir, t1_asl_dir, distcorr_dir, reg_dir, 
                  asl_out_dir, calib_out_dir] ]
        
    # Images required for processing 
    asl = op.join(asl_dir, "tis_stcorr.nii.gz")
    struct = op.join(t1_dir, "T1w_acpc_dc_restore.nii.gz")
    struct_brain = op.join(t1_dir, "T1w_acpc_dc_restore_brain.nii.gz")
    struct_brain_mask = op.join(t1_dir, "T1w_acpc_dc_restore_brain_mask.nii.gz")
    asl_vol0 = op.join(asl_dir, "tis_stcorr_vol1.nii.gz")
    if (not op.exists(asl_vol0) or force_refresh) and target=='asl':
        cmd = "fslroi {} {} 0 1".format(asl, asl_vol0)
        sp.run(cmd.split(" "), check=True)

    # Create ASL-gridded version of T1 image 
    t1_asl_grid = op.join(t1_dir, "ASL", "reg", 
                          "ASL_grid_T1w_acpc_dc_restore.nii.gz")
    if (not op.exists(t1_asl_grid) or force_refresh) and target=='asl':
        asl_spc = rt.ImageSpace(asl)
        t1_spc = rt.ImageSpace(struct)
        t1_asl_grid_spc = t1_spc.resize_voxels(asl_spc.vox_size / t1_spc.vox_size)
        nb.save(
            rt.Registration.identity().apply_to_image(struct, t1_asl_grid_spc), 
            t1_asl_grid)
    
    # Create ASL-gridded version of T1 image
    t1_asl_grid_mask = op.join(reg_dir, "ASL_grid_T1w_acpc_dc_restore_brain_mask.nii.gz")
    if (not op.exists(t1_asl_grid_mask) or force_refresh) and target=='asl':
        asl_spc = rt.ImageSpace(asl)
        t1_spc = rt.ImageSpace(struct_brain)
        t1_asl_grid_spc = t1_spc.resize_voxels(asl_spc.vox_size / t1_spc.vox_size)
        t1_mask = binarise_image(struct_brain)
        t1_mask_asl_grid = rt.Registration.identity().apply_to_array(t1_mask, 
                                                        t1_spc, t1_asl_grid_spc)
        # Re-binarise downsampled mask and save
        t1_asl_grid_mask_array = binary_fill_holes(t1_mask_asl_grid>0.25).astype(np.float32)
        t1_asl_grid_spc.save_image(t1_asl_grid_mask_array, t1_asl_grid_mask) 

    # MCFLIRT ASL using the calibration as reference 
    calib = op.join(sub_base, 'ASL', 'Calib', 'Calib0', 'MTCorr', 'calib0_mtcorr.nii.gz')
    asl = op.join(sub_base, 'ASL', 'TIs', 'tis.nii.gz')
    mcdir = op.join(sub_base, 'ASL', 'TIs', 'MoCo', 'asln2m0.mat')
    asl2calib_mc = rt.MotionCorrection.from_mcflirt(mcdir, asl, calib)

    # Rebase the motion correction to target volume 0 of ASL 
    # The first registration in the series gives us ASL-calibration transform
    calib2asl0 = asl2calib_mc[0].inverse()
    asl_mc = rt.chain(asl2calib_mc, calib2asl0)

    # Generate the gradient distortion correction warp 
    gdc_path = op.join(distcorr_dir, 'fullWarp_abs.nii.gz')
    if (not op.exists(gdc_path) or force_refresh) and target=='asl':
        generate_gdc_warp(asl_vol0, grad_coefficients, distcorr_dir)
    gdc = rt.NonLinearRegistration.from_fnirt(gdc_path, asl_vol0, 
            asl_vol0, intensity_correct=True, constrain_jac=(0.01,100))

    # Stack the cblipped images together for use with topup
    pa_ap_sefms = op.join(distcorr_dir, 'merged_sefms.nii.gz')
    if (not op.exists(pa_ap_sefms) or force_refresh) and target=='asl':
        rt.ImageSpace.save_like(pa_sefm, np.stack((
                                nb.load(pa_sefm).get_data(), 
                                nb.load(ap_sefm).get_data()), 
                                axis=-1), 
                                pa_ap_sefms)
    topup_params = op.join(distcorr_dir, 'topup_params.txt')
    generate_topup_params(topup_params)
    topup_config = "b02b0.cnf"  # Note this file doesn't exist in scope, 
                                # but topup knows where to find it 

    # Generate fieldmaps for use with asl_reg (via topup)
    fmap, fmapmag, fmapmagbrain = [ 
        op.join(distcorr_dir, '{}.nii.gz'.format(s)) 
        for s in [ 'fmap', 'fmapmag', 'fmapmagbrain' ]
    ]          
    if ((not all([ op.exists(p) for p in [fmap, fmapmag, fmapmagbrain] ]))
         or force_refresh) and target=='asl':
        generate_fmaps(pa_ap_sefms, topup_params, topup_config, distcorr_dir)

    # get linear registration from asl to structural
    if target == 'asl':
        unreg_img = asl_vol0
    elif target == 'structural':
        # register perfusion-weighted image to structural instead of asl 0
        unreg_img = op.join(sub_base, "ASL", "TIs", "OxfordASL", 
                            "native_space", "perfusion.nii.gz")
    
    # apply gdc to unreg_img before getting registration to structural
    # only apply to asl_vol0 as perfusion image has already had gdc applied
    distcorr_out_dir = asl_out_dir if target=='structural' else distcorr_dir
    gdc_tis_vol1_name = op.join(distcorr_out_dir, "gdc_tis_vol1.nii.gz")
    if (not op.exists(gdc_tis_vol1_name) or force_refresh) and target=='asl':
        gdc_tis_vol1 = gdc.apply_to_image(src=unreg_img,
                                          ref=unreg_img)
        unreg_img = gdc_tis_vol1_name
        nb.save(gdc_tis_vol1, unreg_img)

    # Initial (linear) asl to structural registration, via first round of asl_reg
    asl2struct_initial_path = op.join(
        reg_dir, 
        'asl2struct_initial_bbr_fsl.mat' if target=='asl' else 'asl2struct_final_bbr_fsl.mat'
    )
    if not op.exists(asl2struct_initial_path) or force_refresh:
        asl2struct_initial_path_temp = op.join(reg_dir, 'asl2struct_initial_bbr_fsl.mat')
        fsdir = op.join(t1_dir, f'{sub_id}_V1_MR')
        generate_asl2struct_initial(asl_vol0, struct, fsdir, reg_dir)
        os.replace(asl2struct_initial_path_temp, asl2struct_initial_path)
    asl2struct_initial = rt.Registration.from_flirt(asl2struct_initial_path, 
                                                    src=unreg_img, ref=struct)

    # Get brain mask in asl space
    if target == 'asl':
        mask_name = op.join(reg_dir, "asl_vol1_mask_init.nii.gz")
    else:
        mask_name = op.join(reg_dir, "asl_vol1_mask_final.nii.gz")
    if not op.exists(mask_name) or force_refresh:
        asl_mask = generate_asl_mask(struct_brain, unreg_img, asl2struct_initial)
        rt.ImageSpace.save_like(unreg_img, asl_mask, mask_name)

    # Brain extract volume 0 of asl series
    gdc_unreg_img_brain = op.join(sub_base, "ASL", "TIs", 
                            "DistCorr", "gdc_tis_vol1_brain.nii.gz")
    if (not op.exists(gdc_unreg_img_brain) or force_refresh) and target=='asl':
        bet(unreg_img, gdc_unreg_img_brain)
        unreg_img = gdc_unreg_img_brain

    # Generate a binary WM mask in the space of the T1 (using FS' aparc+aseg)
    wmmask = op.join(sub_base, "T1w", "wmmask.nii.gz")
    if (not op.exists(wmmask) or force_refresh) and target=='asl':
        aparc_seg = op.join(t1_dir, "aparc+aseg.nii.gz")
        wmmask_img = generate_wmmask(aparc_seg)
        rt.ImageSpace.save_like(struct, wmmask_img, wmmask)

    # Generate the EPI distortion correction warp via asl_reg --final
    epi_dc_path = op.join(
        distcorr_dir,
        'asl2struct_warp_init.nii.gz' if target=='asl' else 'asl2struct_warp_final.nii.gz'
    )
    if not op.exists(epi_dc_path) or force_refresh:
        epi_dc_path_temp = op.join(distcorr_dir, 'asl2struct_warp.nii.gz')
        generate_epidc_warp(unreg_img, struct, struct_brain, 
                            mask_name, wmmask, asl2struct_initial, fmap, 
                            fmapmag, fmapmagbrain, distcorr_dir)
        # rename warp so it isn't overwritten
        os.replace(epi_dc_path_temp, epi_dc_path)
    epi_dc = rt.NonLinearRegistration.from_fnirt(epi_dc_path, 
                mask_name, struct, intensity_correct=True, 
                constrain_jac=(0.01,100))

    # if ending in asl space, chain struct2asl transformation
    if target == 'asl':
        struct2asl_aslreg = op.join(distcorr_out_dir, "struct2asl.mat")
        struct2asl_aslreg = rt.Registration.from_flirt(struct2asl_aslreg,
                                            src=struct, ref=asl)
        epi_dc = rt.chain(epi_dc, struct2asl_aslreg)

    # Final ASL transforms: moco, grad dc, 
    # epi dc (incorporating asl->struct reg)
    asl = op.join(asl_dir, "tis_stcorr.nii.gz")
    reference = t1_asl_grid if target=='structural' else asl
    asl_outpath = op.join(distcorr_out_dir, "tis_distcorr.nii.gz")
    if not op.exists(asl_outpath) or force_refresh:
        asl2struct_mc_dc = rt.chain(asl_mc, gdc, epi_dc)
        asl_corrected = asl2struct_mc_dc.apply_to_image(src=asl, 
                                                        ref=reference, 
                                                        cores=mp.cpu_count())
        nb.save(asl_corrected, asl_outpath)

    # Final calibration transforms: calib->asl, grad dc, 
    # epi dc (incorporating asl->struct reg)
    calib_outpath = op.join(calib_out_dir, "calib0_dcorr.nii.gz")
    if (not op.exists(calib_outpath) or force_refresh) and target=='structural':
        calib2struct_dc = rt.chain(calib2asl0, gdc, epi_dc)
        calib_corrected = calib2struct_dc.apply_to_image(src=calib, 
                                                         ref=reference)
        
        nb.save(calib_corrected, calib_outpath)

    # Final scaling factors transforms: moco, grad dc, 
    # epi dc (incorporating asl->struct reg)
    sfs_name = op.join(asl_dir, "combined_scaling_factors.nii.gz")
    sfs_outpath = op.join(distcorr_out_dir, "combined_scaling_factors.nii.gz")
    if not op.exists(sfs_outpath) or force_refresh:
        # don't chain transformations together if we don't have to
        try:
            asl2struct_mc_dc
        except NameError:
            asl2struct_mc_dc = rt.chain(asl_mc, gdc, epi_dc)
        sfs_corrected = asl2struct_mc_dc.apply_to_image(src=sfs_name, 
                                                        ref=reference, 
                                                        cores=mp.cpu_count())
        nb.save(sfs_corrected, sfs_outpath)

    # create ti image in asl space
    slicedt = 0.059
    tis = [1.7, 2.2, 2.7, 3.2, 3.7]
    sliceband = 10
    ti_asl = op.join(sub_base, "ASL", "TIs", "timing_img.nii.gz")
    if (not op.exists(ti_asl) or force_refresh) and target=='asl':
        create_ti_image(asl, tis, sliceband, slicedt, ti_asl)
    
    # transform ti image into t1 space
    ti_t1 = op.join(t1_asl_dir, "timing_img.nii.gz")
    if (not op.exists(ti_t1) or force_refresh) and target=='structural':
        asl2struct = op.join(distcorr_dir, "asl2struct.mat")
        asl2struct = rt.Registration.from_flirt(asl2struct,
                                                src=asl,
                                                ref=struct)
        ti_t1_img = asl2struct.apply_to_image(src=ti_asl,
                                              ref=reference,
                                              order=0)
        nb.save(ti_t1_img, ti_t1)

if __name__  == '__main__':

    # study_dir = 'HCP_asl_min_req'
    # sub_number = 'HCA6002236'
    # grad_coefficients = 'HCP_asl_min_req/coeff_AS82_Prisma.grad'
    # sys.argv[1:] = ('%s %s -g %s' % (study_dir, sub_number, grad_coefficients)).split()
    main()
