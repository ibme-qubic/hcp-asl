""" 
 Script for gradient and EPI ddistortion correction of HCP ASL data

 Script to generate gradient distortion correction, and EPI distortion
 correction warps for the HCP ASL data, then apply the warps along 
 with motion correction to the ASL-gridded T1w-space, so that all 
 transformations are done in a single interpolation step.

 F. A. Kennedy McConnell - April 2020
"""

import os
import subprocess as sp
import sys
import os.path as op 
import glob 

import regtricks as rt
import nibabel as nb
import numpy as np
from fsl.wrappers import fslmaths

sys.path.append("/mnt/hgfs/shared_with_vm/hcp-asl")

from hcpasl.extract_fs_pvs import extract_fs_pvs
from pathlib import Path
import argparse

# Generate gradient distortion correction warp
def calc_gdc_warp(asldata_vol1, coeffs_loc, oph):
    """
    Generate warp for gradient distortion correction using siemens 
    coefficients file.
    """

    gdc_call = ("gradient_unwarp.py " + asldata_vol1 + " gdc_corr_vol1.nii.gz " +
                    "siemens -g " + coeffs_loc)

    # print(gdc_call)
    sp.run(gdc_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

    gdc_warp_call = ("convertwarp --abs --ref=" + oph + "/gdc_corr_vol1.nii.gz " +
                    "--warp1=" + oph + "/fullWarp_abs.nii.gz --relout --out=" + 
                    oph + "/gdc_warp.nii.gz")

    # print(gdc_warp_call)
    sp.run(gdc_warp_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

def produce_topup_params(pars_filepath):
    """
    Generate a file containing the parameters used by topup to generate EPI distortion
    correction fieldmap and output this into subject's directory. -- This could be a 
    global/config file if that is preferable. If the .txt file already exists, delete 
    and recreate it.
    """
    if os.path.isfile(pars_filepath):
        os.remove(pars_filepath)
    with open(pars_filepath, "a") as t_pars:
        t_pars.write("0 1 0 0.04845" + "\n")
        t_pars.write("0 -1 0 0.04845")
    
def calc_fmaps(pa_sefm, ap_sefm, pa_ap_sefms, pars_filepath, cnf_file, distcorr_dir, out_basename, topup_fmap, 
                fmap_rads, fmapmag, fmapmagbrain):

    """
    Use topup to generate a fieldmap for distortion correction in asl_reg 
    after conversion from Hz to rads/s. Topup also outputs the SEFMs 
    corrected for EPI distortion, which are then available for use as the 
    fieldmap magnitude image and  brain-extracted fieldmap mag image.
    """
    merge_sefm_call = ("fslmerge -t " + pa_ap_sefms + " " + pa_sefm + " " + ap_sefm)
    # print(merge_sefm_call)
    sp.run(merge_sefm_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    
    topup_call = ("topup --imain=" + pa_ap_sefms + 
                    " --datain=" + pars_filepath + " --config=" + cnf_file + 
                    " --out=" + out_basename + " --fout=" +
                    topup_fmap + " --iout=" + distcorr_dir +
                    "/corrected_sefms.nii.gz")
    
    # print(topup_call)
    sp.run(topup_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

    convert_torads = ("fslmaths " + topup_fmap + 
                        " -mul 3.14159 -mul 2 " + "/" + fmap_rads)
    # print(convert_torads)
    sp.run(convert_torads.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

    mean_fmapmag_call = ("fslmaths " + distcorr_dir + "/corrected_sefms.nii.gz -Tmean " + 
                            "/" + fmapmag)
    # print(mean_fmapmag_call)
    sp.run(mean_fmapmag_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

    bet_fmapmag_call = ("bet " + "/" + fmapmag + " /" + fmapmagbrain)
    # print(bet_fmapmag_call)
    sp.run(bet_fmapmag_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

def gen_initial_trans(regfrom, outdir, struct, struct_brain):
    """
    Generate the initial linear transformation between ASL-space and T1w-space
    using asl_reg. This is required as the initalization for the registration
    which generates the distortion correcting warp.
    """
    reg_call = ("asl_reg -i " + regfrom + " -o " + outdir + " -s " + struct +
                " --sbet=" + struct_brain + " --mainonly")
    # print(reg_call)
    sp.run(reg_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

def gen_asl_mask(struct_brain, struct_bet_mask, regfrom, asl2struct, asl_mask,
                struct2asl):
    """
    Generate a mask of the brain in the space of the first ASL volume. The
    make is required for use in asl_reg when it is called for the purpose
    of generating te distortion correction warp.
    """
    invert_reg = ("convert_xfm -omat " + struct2asl + " -inverse " + asl2struct)

    sbrain_call = ("fslmaths " + struct_brain + " -bin " + struct_bet_mask)
    trans_call = ("flirt -in " + struct_bet_mask + " -ref " + regfrom + " -applyxfm -init " + 
                    struct2asl + " -out " + asl_mask + " -interp trilinear -paddingsize 1")
    fill_call = ("fslmaths " + asl_mask + " -thr 0.25 -bin -fillh " + asl_mask)
    hdr_call = ("fslcpgeom " + regfrom + " " + asl_mask)

    # print(invert_reg)
    # print(sbrain_call)
    # print(trans_call)
    # print(fill_call)
    # print(hdr_call)

    sp.run(invert_reg.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    sp.run(sbrain_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    sp.run(trans_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    sp.run(fill_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    sp.run(hdr_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

def gen_pves(t1w_dir, asl, fileroot):
    """
    Generate partial volume estimates from freesurfer segmentations of the cortex
    and subcortical structures.

    Args: 
        t1w_dir: path to subject's T1w directory, containing a T1w scan (eg 
            acdc_dc_restore), aparc+aseg FS volumetric segmentation and 
            fsaverage_32k surface directory
        asl: path to ASL image, used for setting resolution of output 
        fileroot: path basename for output, will add suffix GM/WM/CSF
    """    

    # Load the t1 image, aparc+aseg and surfaces from their expected 
    # names and locations within t1w_dir 
    surf_dict = {}
    t1 = op.join(t1w_dir, 'T1w_acpc_dc.nii.gz')
    aparcseg = op.join(t1w_dir, 'aparc+aseg.nii.gz')
    for k,n in zip(['LWS', 'LPS', 'RPS', 'RWS'], 
                   ['L.white.32k_fs_LR.surf.gii',
                    'L.pial.32k_fs_LR.surf.gii', 
                    'R.pial.32k_fs_LR.surf.gii', 
                    'R.white.32k_fs_LR.surf.gii']):
        paths = glob.glob(op.join(t1w_dir, 'fsaverage_LR32k', '*' + n))
        assert len(paths) == 1, f'Found multiple surfaces for {k}'
        surf_dict[k] = paths[0]

    # Generate a single 4D volume of PV estimates, stacked GM/WM/CSF
    pvs_stacked = extract_fs_pvs(aparcseg, surf_dict, t1, asl)

    # Save output with tissue suffix 
    hdr = pvs_stacked.header 
    aff = pvs_stacked.affine 
    for idx, suffix in enumerate(['GM', 'WM', 'CSF']):
        p = "{}_{}.nii.gz".format(fileroot, suffix)
        nii = nb.Nifti2Image(pvs_stacked.dataobj[...,idx], aff, header=hdr)
        nb.save(nii, p)

def gen_wm_mask(pvwm, tissseg):
    """
    Generate a white matter mask from the WM partial volume estimate for use in
    asl_reg when applied for distortion correction.
    """

    maths_call = ("fslmaths " + pvwm + " -thr 0.5 -bin " + tissseg)
    # print(maths_call)
    sp.run(maths_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    

def calc_distcorr_warp(regfrom, distcorr_dir, struct, struct_brain, mask, tissseg,
                        asl2struct_trans, fmap_rads, fmapmag, fmapmagbrain, asl_grid_T1,
                        gdc_warp):
    """
    Warp is generated from asl_reg, which performs EPI distortion correction of 
    ASL data while transforming the data to alignment with T1w-space. This warp is
    then merged with the gradient distortion correction warp if that is available.
    """

    ## Use asl_reg to inital EPI distcorr warp
    reg_call = ("asl_reg -i " + regfrom + " -o " + distcorr_dir + " -s " +
                struct + " --sbet=" + struct_brain + " -m " + mask + " --tissseg " +
                tissseg + " --imat " + asl2struct_trans + " --finalonly --fmap=" + 
                fmap_rads + " --fmapmag=" + fmapmag + " --fmapmagbrain=" + fmapmagbrain +
                " --pedir=y --echospacing=0.00057")
    # print(reg_call)
    sp.run(reg_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    
    # Add the gradient distortion correction warp to the EPI distortion correction warp
    if os.path.isfile(gdc_warp):
        merge_warp_call = ("convertwarp -r " + asl_grid_T1 + " -o " + distcorr_dir + 
                        "/distcorr_warp -w " + distcorr_dir + "/asl2struct_warp --warp2=" + 
                        gdc_warp + " --rel")
        # print(merge_warp_call)
        sp.run(merge_warp_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    else:
        print("Gradient distortion correction not applied")
        cp_call = ("imcp " + distcorr_dir + "/asl2struct_warp.nii.gz " + distcorr_dir +
                    "/distcorr_warp")
        # print(cp_call)
        sp.run(cp_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

# calculate the jacobian of the warp for intensity correction
def calc_warp_jacobian(distcorr_dir):
    """
    Calculation of the Jacobian of the combined distortion correction for subsequent 
    Jacobian intensity scaling
    """
    utils_call1 = ("fnirtfileutils -i " + distcorr_dir + "/distcorr_warp -f spline -o " + 
                    distcorr_dir + "/distcorr_warp_coeff")
    utils_call2 = ("fnirtfileutils -i " + distcorr_dir + "/distcorr_warp_coeff -j " +
                    distcorr_dir + "/distcorr_jacobian")
    hdr_call = ("fslcpgeom " + distcorr_dir + "/distcorr_warp " + distcorr_dir + 
                    "/distcorr_jacobian -d")

    # print(utils_call1)
    # print(utils_call2)
    # print(hdr_call)

    sp.run(utils_call1.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    sp.run(utils_call2.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    sp.run(hdr_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

# apply the combined distortion correction warp
def apply_distcorr_warp(asldata_orig, T1space_ref, asldata_T1space, distcorr_dir,
                        moco_xfms, calib_orig, calib_T1space, calib_xfms, sfacs_orig=None,
                        sfacs_T1space=None):
    """
    Application of motion correction, and EPI and gradient distortion corrections, 
    and transformation to T1w-space in one step. Leaving the ASL and calibration data 
    in ASL-gridded T1w-space.
    """

    asl_apply_call = ("applywarp -i " + asldata_orig + " -r " + T1space_ref + " -o " +
                    asldata_T1space + " --premat=" + moco_xfms + " -w " + 
                    distcorr_dir + "/distcorr_warp" + " --rel --interp=trilinear" +
                    " --paddingsize=1 --super --superlevel=a")

    asl_jaco_call = ("fslmaths " + asldata_T1space + " -mul " + distcorr_dir + 
                    "/distcorr_jacobian " + asldata_T1space)

    calib_apply_call = ("applywarp -i " + calib_orig + " -r " + T1space_ref + " -o " +
                    calib_T1space + " --premat=" + calib_xfms + " -w " + 
                    distcorr_dir + "/distcorr_warp" + " --rel --interp=trilinear" +
                    " --paddingsize=1 --super --superlevel=a")

    calib_jaco_call = ("fslmaths " + calib_T1space + " -mul " + distcorr_dir + 
                    "/distcorr_jacobian " + calib_T1space)
    # print(asl_apply_call)
    # print(asl_jaco_call)
    # print(calib_apply_call)
    # print(calib_jaco_call)
    sp.run(asl_apply_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    sp.run(asl_jaco_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    sp.run(calib_apply_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    sp.run(calib_jaco_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    if (sfacs_orig and sfacs_T1space):
        sfacs_apply_call = ("applywarp -i " + sfacs_orig + " -r " + T1space_ref + " -o " +
                        sfacs_T1space + " --premat=" + moco_xfms + " -w " + 
                        distcorr_dir + "/distcorr_warp" + " --rel --interp=trilinear" +
                        " --paddingsize=1 --super --superlevel=a")

        sfacs_jaco_call = ("fslmaths " + sfacs_T1space + " -mul " + distcorr_dir + 
                        "/distcorr_jacobian " + sfacs_T1space)
        # print(sfacs_apply_call)
        # print(sfacs_jaco_call)
        sp.run(sfacs_apply_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        sp.run(sfacs_jaco_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

def find_field_maps(study_dir, subject_number):
    """
    Find the mbPCASL field maps in the subject's directory.
    The field maps are found in the subject's B session directory. 
    Multiple pairs of field maps are taken in the B session; this 
    function assumes that the mbPCASL field maps are the final 2 
    field map directories in the session.
    """
    scan_dir = Path(study_dir) / subject_number / f'{subject_number}_V1_B/scans'
    fm_dirs = sorted(scan_dir.glob('**/*-FieldMap_SE_EPI'))[-2:]
    if (fm_dirs[0] / f'resources/NIFTI/files/{subject_number}_V1_B_PCASLhr_SpinEchoFieldMap_PA.nii.gz').exists():
        pa_dir, ap_dir = fm_dirs
    elif (fm_dirs[1] / f'resources/NIFTI/files/{subject_number}_V1_B_PCASLhr_SpinEchoFieldMap_PA.nii.gz').exists():
        ap_dir, pa_dir = fm_dirs
    pa_sefm = pa_dir / f'resources/NIFTI/files/{subject_number}_V1_B_PCASLhr_SpinEchoFieldMap_PA.nii.gz'
    ap_sefm = ap_dir / f'resources/NIFTI/files/{subject_number}_V1_B_PCASLhr_SpinEchoFieldMap_AP.nii.gz'
    return str(pa_sefm), str(ap_sefm)
    
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
        "--nobc",
        default=False,
        action='store_true',
        help="Use this flag to run without banding correction steps."
    )
    args = parser.parse_args()
    study_dir = args.study_dir
    sub_num = args.sub_number
    grad_coeffs = args.grads

    if args.nobc:
        ext = "_nobc"
    else:
        ext = None
    oph = (study_dir + "/" + sub_num + f"/ASL/TIs/DistCorr{ext}")
    outdir = (study_dir + "/" + sub_num + f"/T1w/ASL/reg{ext}")
    pve_path = (study_dir + "/" + sub_num + f"/T1w/ASL/PVEs{ext}")
    T1w_oph = (study_dir + "/" + sub_num + f"/T1w/ASL/TIs{ext}/DistCorr{ext}")
    T1w_cal_oph  = (study_dir + "/" + sub_num + f"/T1w/ASL/Calib{ext}/Calib0/DistCorr{ext}")
    need_dirs = [oph, outdir, pve_path, T1w_oph, T1w_cal_oph]
    for req_dir in need_dirs:
        Path(req_dir).mkdir(parents=True, exist_ok=True)

    initial_wd = os.getcwd()
    print("Pre-distortion correction working directory was: " + initial_wd)
    print("Changing working directory to: " + oph)
    os.chdir(oph)

    # Generate ASL-gridded T1-aligned T1w image for use as a reg reference
    t1 = (study_dir + "/" + sub_num + "/T1w/T1w_acpc_dc_restore.nii.gz")
    t1_brain = (study_dir + "/" + sub_num + "/T1w/T1w_acpc_dc_restore_brain.nii.gz")

    if args.nobc:
        asl = (study_dir + "/" + sub_num + "/ASL/TIs/BiasCorr/tis_biascorr.nii.gz")
    else:
        asl = (study_dir + "/" + sub_num + "/ASL/TIs/STCorr/SecondPass/tis_stcorr.nii.gz") 
    t1_asl_res = (outdir + "/ASL_grid_T1w_acpc_dc_restore.nii.gz")

    asl_v1 = (asl.split(".")[0] + "_vol1.nii.gz")
    first_asl_call = ("fslroi " + asl + " " + asl_v1 + " 0 1")
    # print(first_asl_call)
    sp.run(first_asl_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

    print("Running regtricks bit")
    t1_spc = rt.ImageSpace(t1)
    asl_spc = rt.ImageSpace(asl_v1)
    t1_spc_asl = t1_spc.resize_voxels(asl_spc.vox_size / t1_spc.vox_size)
    r = rt.Registration.identity()
    t1_asl = r.apply_to_image(t1, t1_spc_asl)
    nb.save(t1_asl, t1_asl_res)
    # Check .grad coefficients are available and call function to generate 
    # GDC warp if they are:
    if os.path.isfile(grad_coeffs):
        calc_gdc_warp(asl_v1, grad_coeffs, oph)
    else:
        print("Gradient coefficients not available")

    print("Changing back to original working directory: " + initial_wd)
    os.chdir(initial_wd)

    # output file of topup parameters to subject's distortion correction dir
    pars_filepath = (oph + "/topup_params.txt")
    produce_topup_params(pars_filepath)

    # generate EPI distortion correction fieldmaps for use in asl_reg
    pa_sefm, ap_sefm = find_field_maps(study_dir, sub_num)
    pa_ap_sefms = (oph + "/merged_sefms.nii.gz")
    cnf_file = "b02b0.cnf"
    out_basename = (oph + "/topup_result")
    topup_fmap = (oph + "/topup_result_fmap_hz.nii.gz")
    fmap_rads = (oph + "/fmap_rads.nii.gz")
    fmapmag = (oph + "/fmapmag.nii.gz")
    fmapmagbrain = (oph + "/fmapmag_brain.nii.gz")
    calc_fmaps(pa_sefm, ap_sefm, pa_ap_sefms, pars_filepath, cnf_file, oph, out_basename, topup_fmap, 
                fmap_rads, fmapmag, fmapmagbrain)
    
    # Calculate initial linear transformation from ASL-space to T1w-space
    asl_v1_brain = (asl_v1.split(".")[0] + "_brain.nii.gz")
    bet_regfrom_call = ("bet " + asl_v1 + " " + asl_v1_brain)
    # print(bet_regfrom_call)
    sp.run(bet_regfrom_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

    gen_initial_trans(asl_v1_brain, outdir, t1, t1_brain)

    # Generate a brain mask in the space of the 1st ASL volume
    asl2struct = (outdir + "/asl2struct.mat")
    t1_brain_mask = (outdir + "/T1w_acpc_dc_restore_brain_mask.nii.gz")
    asl_mask = (outdir + "/asl_vol1_mask.nii.gz")
    struct2asl = (outdir + "/struct2asl.mat")

    gen_asl_mask(t1_brain, t1_brain_mask, asl_v1_brain, asl2struct, asl_mask,
                struct2asl)

    # brain mask
    t1_mask = (outdir + "/T1w_acpc_dc_restore_brain_mask.nii.gz")
    t1_asl_mask_name = (outdir + "/ASL_grid_T1w_acpc_dc_restore_brain_mask.nii.gz")
    t1_mask_spc = rt.ImageSpace(t1_mask)
    t1_mask_spc_asl = t1_mask_spc.resize_voxels(asl_spc.vox_size / t1_mask_spc.vox_size)
    r = rt.Registration.identity()
    t1_mask_asl = r.apply_to_image(t1_mask, t1_mask_spc_asl)
    fslmaths(t1_mask_asl).thr(0.5).bin().run(t1_asl_mask_name)
    
    # Generate PVEs
    aparc_aseg = (study_dir + "/" + sub_num + "/T1w/aparc+aseg.nii.gz")
    pve_files = (pve_path + "/pve")
    gen_pves(Path(t1).parent, asl, pve_files)

    # Generate WM mask
    pvwm = (pve_files + "_WM.nii.gz")
    tissseg = (pve_path + "/wm_mask.nii.gz")
    gen_wm_mask(pvwm, tissseg)
    

    # Calculate the overall distortion correction warp
    asl2str_trans = (outdir + "/asl2struct.mat")
    gdc_warp = (oph + "/gdc_warp.nii.gz")
    calc_distcorr_warp(asl_v1_brain, oph, t1, t1_brain, asl_mask, tissseg,
                        asl2str_trans, fmap_rads, fmapmag, fmapmagbrain, t1_asl_res,
                        gdc_warp)

    # Calculate the Jacobian of the distortion correction warp 
    calc_warp_jacobian(oph)

    # apply the combined distortion correction warp with motion correction
    # to move asl data, calibrationn images, and scaling factors into 
    # ASL-gridded T1w-aligned space
    
    asl_distcorr = (T1w_oph + "/tis_distcorr.nii.gz")
    moco_xfms = (study_dir + "/" + sub_num + f"/ASL/TIs/MoCo{ext}/asln2asl0.mat") #will this work?
    concat_xfms = str(Path(moco_xfms).parent / f'{Path(moco_xfms).stem}.cat')
    # concatenate xfms like in oxford_asl
    concat_call = f'cat {moco_xfms}/MAT* > {concat_xfms}'
    sp.run(concat_call, shell=True)
    # only correcting and transforming the 1st of the calibration images at the moment
    if args.nobc:
        calib_orig = (study_dir + "/" + sub_num + "/ASL/Calib/Calib0/BiasCorr/calib0_restore.nii.gz")
        sfacs_orig = None
        sfacs_distcorr = None
    else:
        calib_orig = (study_dir + "/" + sub_num + "/ASL/Calib/Calib0/MTCorr/calib0_mtcorr.nii.gz")
        sfacs_orig = (study_dir + "/" + sub_num + "/ASL/TIs/STCorr/SecondPass/combined_scaling_factors.nii.gz")
        sfacs_distcorr = (T1w_oph + "/combined_scaling_factors.nii.gz")
    calib_distcorr = (T1w_cal_oph + "/calib0_dcorr.nii.gz")
    calib_inv_xfm = (study_dir + "/" + sub_num + f"/ASL/TIs/MoCo{ext}/asln2m0.mat/MAT_0000")
    calib_xfm = (study_dir + "/" + sub_num + f"/ASL/TIs/MoCo{ext}/calibTOasl1.mat")

    invert_call = ("convert_xfm -omat " + calib_xfm + " -inverse " + calib_inv_xfm)
    # print(invert_call)
    sp.run(invert_call.split(), check=True, stderr=sp.PIPE, stdout=sp.PIPE)

    apply_distcorr_warp(asl, t1_asl_res, asl_distcorr, oph,
                        concat_xfms, calib_orig, calib_distcorr, calib_xfm, sfacs_orig,
                        sfacs_distcorr)

    


if __name__ == "__main__":
    main()