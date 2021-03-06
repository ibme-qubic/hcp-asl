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

from hcpasl.distortion_correction import (
    generate_gdc_warp, generate_topup_params, generate_fmaps, 
    generate_epidc_warp, register_fmap
)
from hcpasl.m0_mt_correction import generate_asl2struct

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
    mask = (image.data>threshold).astype(np.float32)
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

def register_fmap(fmapmag, fmapmagbrain, s, sbet, out_dir, wm_tissseg):
    # create output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # get schedule
    fsldir = os.environ.get('FSLDIR')
    schedule = Path(fsldir)/'etc/flirtsch/bbr.sch'

    # set up commands
    init_xform = out_dir/'fmapmag2struct_init.mat'
    sec_xform = out_dir/'fmapmag2struct_sec.mat'
    bbr_xform = out_dir/'fmapmag2struct_bbr.mat'
    init_cmd = [
        'flirt',
        '-in', fmapmagbrain,
        '-ref', sbet,
        '-dof', '6',
        '-omat', init_xform
    ]
    sec_cmd = [
        'flirt',
        '-in', fmapmag,
        '-ref', s,
        '-dof', '6',
        '-init', init_xform,
        '-omat', sec_xform,
        '-nosearch'
    ]
    bbr_cmd = [
        'flirt',
        '-ref', s,
        '-in', fmapmag,
        '-dof', '6',
        '-cost', 'bbr',
        '-wmseg', wm_tissseg,
        '-init', sec_xform,
        '-omat', bbr_xform,
        '-schedule', schedule
    ]
    for cmd in (init_cmd, sec_cmd, bbr_cmd):
        sp.run(cmd, check=True)
    return str(bbr_xform)

def main():

    # argument handling
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study_dir",
        help="Path of the base study directory.",
        required=True
    )
    parser.add_argument(
        "--sub_id",
        help="Subject number.",
        required=True
    )
    parser.add_argument(
        "-g",
        "--grads",
        help="Filename of the gradient coefficients for gradient"
            + "distortion correction.",
        required=True
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
        help="Filename for the AP fieldmap for use in distortion correction",
        required=True
    )
    parser.add_argument(
        "--fmap_pa",
        help="Filename for the PA fieldmap for use in distortion correction",
        required=True
    )
    parser.add_argument(
        '--use_t1',
        help="If this flag is provided, the T1 estimates from the satrecov "
            + "will also be registered to ASL-gridded T1 space for use in "
            + "perfusion estimation via oxford_asl.",
        action='store_true'
    )
    parser.add_argument(
        "--mtname",
        help="Filename of the empirically estimated MT-correction"
            + "scaling factors.",
        default=None,
        required=not "--nobandingcorr" in sys.argv
    )
    parser.add_argument(
        "-c",
        "--cores",
        help="Number of cores to use when applying motion correction and "
            +"other potentially multi-core operations. Default is the "
            +f"number of cores your machine has ({mp.cpu_count()}).",
        default=mp.cpu_count(),
        type=int,
        choices=range(1, mp.cpu_count()+1)
    )
    parser.add_argument(
        "--interpolation",
        help="Interpolation order for registrations. This can be any "
            +"integer from 0-5 inclusive. Default is 3. See scipy's "
            +"map_coordinates for more details.",
        default=3,
        type=int,
        choices=range(0, 5+1)
    )
    parser.add_argument(
        "--nobandingcorr",
        help="If this option is provided, the MT and ST banding corrections "
            +"won't be applied. This is to be used to compare the difference "
            +"our banding corrections make.",
        action="store_true"
    )
    parser.add_argument(
        "--outdir",
        help="Name of the directory within which we will store all of the "
            +"pipeline's outputs in sub-directories. Default is 'hcp_asl'",
        default="hcp_asl"
    )
    args = parser.parse_args()
    study_dir = args.study_dir
    sub_id = args.sub_id
    grad_coefficients = args.grads
    target = args.target
    pa_sefm = args.fmap_pa
    ap_sefm = args.fmap_ap
    use_t1 = args.use_t1
    mt_factors = args.mtname

    # For debug, re-use existing intermediate files 
    force_refresh = True

    # Input, output and intermediate directories
    # Create if they do not already exist. 
    sub_base = op.abspath(op.join(study_dir, sub_id))
    grad_coefficients = op.abspath(grad_coefficients)
    t1_asl_dir = op.join(sub_base, args.outdir, "ASLT1w")
    distcorr_dir = op.join(sub_base, args.outdir, "ASL", "TIs", "DistCorr")
    reg_dir = op.join(t1_asl_dir, 'reg')
    pvs_dir = op.join(t1_asl_dir, "PVEs")
    t1_dir = op.join(sub_base, f"{sub_id}_V1_MR", "resources", 
                     "Structural_preproc", "files", f"{sub_id}_V1_MR","T1w")
    asl_dir = op.join(sub_base, args.outdir, "ASL", "TIs", "STCorr2") if not args.nobandingcorr else op.join(sub_base, args.outdir, "ASL", "TIs", "MoCo")
    asl_out_dir = op.join(t1_asl_dir, "TIs", "DistCorr")
    calib_out_dir = op.join(t1_asl_dir, "Calib", "Calib0", "DistCorr") if target=='structural' else op.join(sub_base, args.outdir, "ASL", "Calib", "Calib0", "DistCorr")
    [ os.makedirs(d, exist_ok=True) 
        for d in [pvs_dir, t1_asl_dir, distcorr_dir, reg_dir, 
                  asl_out_dir, calib_out_dir] ]
        
    # Images required for processing 
    asl = op.join(asl_dir, "tis_stcorr.nii.gz")if not args.nobandingcorr else op.join(asl_dir, "reg_gdc_dc_tis_biascorr.nii.gz")
    struct = op.join(t1_dir, "T1w_acpc_dc_restore.nii.gz")
    struct_brain = op.join(t1_dir, "T1w_acpc_dc_restore_brain.nii.gz")
    struct_brain_mask = op.join(t1_dir, "brainmask_fs.nii.gz")
    asl_vol0 = op.join(asl_dir, "tis_vol1.nii.gz")
    if (not op.exists(asl_vol0) or force_refresh) and target=='asl':
        cmd = "fslroi {} {} 0 1".format(asl, asl_vol0)
        sp.run(cmd.split(" "), check=True)

    # Create ASL-gridded version of T1 image 
    t1_asl_grid = op.join(reg_dir, "ASL_grid_T1w_acpc_dc_restore.nii.gz")
    if (not op.exists(t1_asl_grid) or force_refresh) and target=='asl':
        asl_spc = rt.ImageSpace(asl)
        t1_spc = rt.ImageSpace(struct)
        t1_asl_grid_spc = t1_spc.resize_voxels(asl_spc.vox_size / t1_spc.vox_size)
        nb.save(
            rt.Registration.identity().apply_to_image(struct, 
                                                      t1_asl_grid_spc, 
                                                      order=args.interpolation), 
            t1_asl_grid
        )
    
    # Create ASL-gridded version of T1 image
    t1_asl_grid_mask = op.join(reg_dir, "ASL_grid_T1w_acpc_dc_restore_brain_mask.nii.gz")
    if (not op.exists(t1_asl_grid_mask) or force_refresh) and target=='asl':
        asl_spc = rt.ImageSpace(asl)
        t1_spc = rt.ImageSpace(struct_brain)
        t1_asl_grid_spc = t1_spc.resize_voxels(asl_spc.vox_size / t1_spc.vox_size)
        t1_mask = nb.load(struct_brain_mask).get_fdata()
        t1_mask_asl_grid = rt.Registration.identity().apply_to_array(t1_mask, 
                                                                     t1_spc, 
                                                                     t1_asl_grid_spc, 
                                                                     order=0)
        # Re-binarise downsampled mask and save
        t1_asl_grid_mask_array = binary_fill_holes(t1_mask_asl_grid>0.25).astype(np.float32)
        t1_asl_grid_spc.save_image(t1_asl_grid_mask_array, t1_asl_grid_mask) 

    # MCFLIRT ASL using the calibration as reference 
    calib = op.join(sub_base, args.outdir, 'ASL', 'Calib', 'Calib0', 'calib0.nii.gz')
    asl = op.join(sub_base, args.outdir, 'ASL', 'TIs', 'tis.nii.gz')
    mcdir = op.join(sub_base, args.outdir, 'ASL', 'TIs', 'MoCo', 'asln2m0.mat')
    asl2calib_mc = rt.MotionCorrection.from_mcflirt(mcdir, asl, calib)

    # Rebase the motion correction to target volume 0 of ASL 
    # The first registration in the series gives us ASL-calibration transform
    calib2asl0 = asl2calib_mc[0].inverse()
    asl_mc = rt.chain(asl2calib_mc, calib2asl0)

    # load the gradient distortion correction warp 
    gdc_path = op.join(sub_base, args.outdir, "ASL", "gradient_unwarp", "fullWarp_abs.nii.gz")
    gdc = rt.NonLinearRegistration.from_fnirt(gdc_path, asl_vol0, 
            asl_vol0, intensity_correct=True, constrain_jac=(0.01,100))

    # get fieldmap names for use with asl_reg
    fmap, fmapmag, fmapmagbrain = [ 
        op.join(sub_base, args.outdir, "ASL", "topup", '{}.nii.gz'.format(s)) 
        for s in [ 'fmap', 'fmapmag', 'fmapmagbrain' ]
    ]
    
    # load the epi distortion correction warp from topup
    dc_path = op.join(sub_base, args.outdir, "ASL", "topup", "WarpField_01.nii.gz")
    dc_warp = rt.NonLinearRegistration.from_fnirt(coefficients=dc_path,
                                                  src=fmapmag,
                                                  ref=fmapmag,
                                                  intensity_correct=True,
                                                  constrain_jac=(0.01, 100))

    # get linear registration from asl to structural
    if target == 'asl':
        unreg_img = asl_vol0
    elif target == 'structural':
        # register perfusion-weighted image to structural instead of asl 0
        unreg_img = op.join(sub_base, args.outdir, "ASL", "TIs", "OxfordASL", 
                            "native_space", "perfusion.nii.gz")
    
    # set correct output directory
    distcorr_out_dir = asl_out_dir if target=='structural' else distcorr_dir

    # get asl to structural registration, via bbregister
    # only need this if target space == structural
    asl2struct_path = op.join(reg_dir, 'asl2struct.mat')
    if (not op.exists(asl2struct_path) or force_refresh) and target=='structural':
        fsdir = op.join(t1_dir, f"{sub_id}_V1_MR")
        generate_asl2struct(unreg_img, struct, fsdir, reg_dir)
    if target == 'structural':
        asl2struct_reg = rt.Registration.from_flirt(asl2struct_path, 
                                                        src=unreg_img, 
                                                        ref=struct)
    elif target == 'asl':
        calib2struct_name = op.join(calib_out_dir, "asl2struct.mat")
        calib2struct = rt.Registration.from_flirt(calib2struct_name,
                                                  calib,
                                                  struct_brain)
        asl2struct_reg = rt.chain(calib2asl0.inverse(), calib2struct)

    # Get brain mask in asl space for use with oxford_asl later
    mask_name = op.join(reg_dir, "asl_vol1_mask_init.nii.gz")
    if (not op.exists(mask_name) or force_refresh) and target=="asl":
        asl_mask = asl2struct_reg.inverse().apply_to_image(struct_brain_mask,
                                                               unreg_img, 
                                                               order=0)
        asl_mask = nb.nifti1.Nifti1Image(np.where(asl_mask.get_fdata()>0.25, 1., 0.),
                                         affine=asl_mask.affine)
        nb.save(asl_mask, mask_name)

    # Final ASL transforms: moco, grad dc, 
    # epi dc (incorporating asl->struct reg)
    reference = t1_asl_grid if target=='structural' else asl
    asl_outpath = op.join(distcorr_out_dir, "tis_distcorr.nii.gz")
    if (not op.exists(asl_outpath) or force_refresh) and target=='structural':
        asl = op.join(sub_base, args.outdir, "ASL", "TIs", "tis.nii.gz")
        asl2struct_mc_dc = rt.chain(gdc, dc_warp, asl_mc, asl2struct_reg)
        asl_corrected = asl2struct_mc_dc.apply_to_image(src=asl, 
                                                        ref=reference, 
                                                        cores=args.cores,
                                                        order=args.interpolation)
        nb.save(asl_corrected, asl_outpath)

    # Final calibration transforms: calib->asl, grad dc, 
    # epi dc (incorporating asl->struct reg)
    calib_outpath = op.join(calib_out_dir, "calib0_dcorr.nii.gz")
    if (not op.exists(calib_outpath) or force_refresh) and target=='structural':
        calib = op.join(sub_base, args.outdir, "ASL", "Calib", "Calib0", "calib0.nii.gz")
        calib2struct_dc = rt.chain(gdc, dc_warp, calib2asl0, asl2struct_reg)
        calib_corrected = calib2struct_dc.apply_to_image(src=calib, 
                                                         ref=reference,
                                                         order=args.interpolation)
        
        nb.save(calib_corrected, calib_outpath)

    # apply registrations to fmapmag.nii.gz
    if target=='structural':
        fmap_struct_dir = op.join(sub_base, args.outdir, "ASL", "topup", "fmap_struct_reg")
        fmap2struct_bbr = rt.Registration.from_flirt(op.join(fmap_struct_dir, "asl2struct.mat"),
                                                     src=str(fmapmag),
                                                     ref=str(struct))
        fmap_struct = fmap2struct_bbr.apply_to_image(src=fmapmag, ref=reference)
        fmap_struct_name = op.join(fmap_struct_dir, "fmapmag_aslstruct.nii.gz")
        nb.save(fmap_struct, fmap_struct_name)
    
    # apply registrations to satrecov-estimated T1 image for use with oxford_asl
    reg_est_t1_name = op.join(reg_dir, "mean_T1t_filt.nii.gz")
    if (not op.exists(reg_est_t1_name) or force_refresh) and target=='structural' and use_t1:
        est_t1_name = op.join(sub_base, args.outdir, "ASL", "TIs", "SatRecov2", 
                                "spatial", "mean_T1t_filt.nii.gz")
        reg_est_t1 = asl2struct_reg.apply_to_image(src=est_t1_name,
                                                   ref=reference,
                                                   order=args.interpolation)
        nb.save(reg_est_t1, reg_est_t1_name)

    # create ti image in asl space
    slicedt = 0.059
    tis = [1.7, 2.2, 2.7, 3.2, 3.7]
    sliceband = 10
    ti_asl = op.join(sub_base, args.outdir, "ASL", "TIs", "timing_img.nii.gz")
    if (not op.exists(ti_asl) or force_refresh) and target=='asl':
        create_ti_image(asl, tis, sliceband, slicedt, ti_asl)
    
    # transform ti image into t1 space
    ti_t1 = op.join(t1_asl_dir, "timing_img.nii.gz")
    if (not op.exists(ti_t1) or force_refresh) and target=='structural':
        ti_t1_img = asl2struct_reg.apply_to_image(src=ti_asl,
                                                  ref=reference,
                                                  order=0)
        nb.save(ti_t1_img, ti_t1)

    # register scaling factors to ASL-gridded T1 space
    if not args.nobandingcorr:
        # apply calib->structural registration to mt scaling factors
        mt_sfs_calib_name = op.join(calib_out_dir, "mt_scaling_factors_calibstruct.nii.gz")
        if (not op.exists(mt_sfs_calib_name) or force_refresh) and target=='structural':
            # create MT scaling factor image in calibration image space
            calib_img = nb.load(calib)
            mt_sfs = np.loadtxt(mt_factors)
            mt_img = nb.nifti1.Nifti1Image(np.tile(mt_sfs, (86, 86, 1)),
                                        affine=calib_img.affine)
            calib2struct = rt.chain(calib2asl0, asl2struct_reg)
            mt_calibstruct_img = calib2struct.apply_to_image(src=mt_img,
                                                            ref=reference,
                                                            order=args.interpolation)
            nb.save(mt_calibstruct_img, mt_sfs_calib_name)

    # Final scaling factors transforms: moco, grad dc, 
    # epi dc (incorporating asl->struct reg)
    sfs_name = op.join(asl_dir, "combined_scaling_factors.nii.gz")
    sfs_outpath = op.join(distcorr_out_dir, "combined_scaling_factors.nii.gz")
    if (not op.exists(sfs_outpath) or force_refresh) and target=="structural":
        sfs_corrected = asl2struct_reg.apply_to_image(src=sfs_name, 
                                                    ref=reference, 
                                                    cores=args.cores)
        nb.save(sfs_corrected, sfs_outpath)

if __name__  == '__main__':

    # study_dir = 'HCP_asl_min_req'
    # sub_number = 'HCA6002236'
    # grad_coefficients = 'HCP_asl_min_req/coeff_AS82_Prisma.grad'
    # sys.argv[1:] = ('%s %s -g %s' % (study_dir, sub_number, grad_coefficients)).split()
    main()
