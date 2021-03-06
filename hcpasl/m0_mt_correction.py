"""
This contains a range of functions required to correct for the 
Magnetisation Transfer effect visible in the HCP data.
"""

import json
from pathlib import Path
from fsl.wrappers import fslmaths, LOAD, bet, fast
from fsl.data.image import Image
import numpy as np
from .initial_bookkeeping import create_dirs
from .tissue_masks import generate_tissue_mask
from .distortion_correction import register_fmap, generate_asl_mask
import subprocess
import regtricks as rt
import nibabel as nb
import pandas as pd

import os
import os.path as op

def generate_asl2struct(asl_vol0, struct, fsdir, reg_dir):
    """
    Generate the linear transformation between ASL-space and T1w-space
    using FS bbregister. Note that struct is required only for saving 
    the output in the right convention, it is not actually used by 
    bbregister. 
    
    Args:
        asl_vol0: path to first volume of ASL 
        struct: path to T1w image (eg T1w_acdc_restore.nii.gz)
        fsdir: path to subject's FreeSurfer output directory 
        reg_dir: path to registration directory, for output 

    Returns: 
        n/a, file 'asl2struct.mat' will be saved in reg_dir
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
    omat_path = op.join(reg_dir, "asl2struct.mat")
    cmd = os.environ['SUBJECTS_DIR'] = new_sd
    cmd = f"$FREESURFER_HOME/bin/bbregister --s {sid} --mov {asl_vol0} --t2 "
    cmd += f"--reg asl2orig_mgz_initial_bbr.dat --fslmat {omat_path} --init-fsl"
    subprocess.run(cmd, shell=True)

    try:
        asl2orig_fsl = rt.Registration.from_flirt(str(omat_path), str(asl_vol0), str(orig_mgz))
    except RuntimeError as e:
        # final row != [0 0 0 1], round to 5 d.p. and try again
        print(e)
        print("Rounding to 5 d.p.")
        arr = np.loadtxt(omat_path)
        np.savetxt(omat_path, arr, fmt='%.5f')
        asl2orig_fsl = rt.Registration.from_flirt(str(omat_path), str(asl_vol0), str(orig_mgz))

    # Return to original working directory, and flip the FSL matrix to target
    # asl -> T1, not orig.mgz. Save output. 
    os.chdir(pwd)
    if old_sd:
        os.environ['SUBJECTS_DIR'] = old_sd
    asl2struct_fsl = asl2orig_fsl.to_flirt(str(asl_vol0), str(struct))
    np.savetxt(op.join(reg_dir, 'asl2struct.mat'), asl2struct_fsl)

def load_json(subject_dir):
    """
    Load json but with some error-checking to make sure it exists.
    If it doesn't exist, instruct user to run the first part of 
    the pipeline first.

    Parameters
    ----------
    subject_dir : pathlib.Path
        Path to subject's base directory.
    
    Returns
    -------
    dict
        Dictionary containing important file paths.
    """
    json_name = subject_dir / 'ASL/ASL.json'
    if json_name.exists():
        with open(json_name, 'r') as infile:
            json_dict = json.load(infile)
    else:
        raise Exception(f'File {json_name} does not exist.' + 
                        'Please run initial_processing() first.')
    return json_dict

def update_json(new_dict, old_dict):
    """
    Add key-value pairs to the dictionary.

    Add the key-value pairs in `new_dict` to `old_dict` 
    and save the resulting dictionary to the json found in 
    `old_dict['json_name']`.

    Parameters
    ----------
    new_dict : dict
        New key-value pairs to add to the json of important 
        file names.
    old_dict : dict
        Dictionary to be updated. Also has a field containing 
        the location of the json to update.
    """
    old_dict.update(new_dict)
    with open(Path(old_dict['json_name']), 'w') as fp:
        json.dump(old_dict, fp, sort_keys=True, indent=4)

def parse_LUT(LUT_name):
    """
    Parse a FreeSurfer Lookup-Table returning the desired label 
    names.

    Parameters
    ----------
    LUT_name : str
        Path to the Lookup-Table
    
    Returns
    -------
    labels : list of ints
        List of int labels from the LUT
    """
    lut = pd.read_csv(LUT_name, header=None)
    labels = [int(row[0].split(" ")[0]) for _, row in lut[1::2].iterrows()]
    return labels

def correct_M0(subject_dir, mt_factors, wmparc, ribbon, 
               corticallut, subcorticallut, interpolation=3,
               nobandingcorr=False, outdir="hcp_asl"):
    """
    Correct the M0 images.
    
    For each of the subject's two calibration images:
    #. Use BET on the image;
    #. Use FAST on the brain-extracted image to obtain the bias-field;
    #. Perform bias correction;
    #. Multiply by the provided 'mt_factors' for MT-effect correction.
    
    Parameters
    ----------
    subject_dir : pathlib.Path
        Path to the subject's base directory.
    mt_factors : pathlib.Path
        Path to the empirically estimated MT correction 
        scaling factors.
    nobandingcorr : bool, optional
        If this is True, the banding correction options in the 
        pipeline will be switched off. Default is False (i.e. 
        banding corrections are applied by default).
    outdir : str
        Name of the main results directory. Default is 'hcp_asl'.
    """
    # load json containing info on where files are stored
    json_dict = load_json(subject_dir/outdir)
    
    # do for both m0 images for the subject, calib0 and calib1
    calib_names = [json_dict['calib0_img'], json_dict['calib1_img']]

    # get structural image names
    struct_name, struct_brain_name = [json_dict[key] for key in ("T1w_acpc", "T1w_acpc_brain")]

    # generate white matter mask in T1w space for use in registration
    t1reg_dir = Path(json_dict["structasl"])/"reg"
    t1reg_dir.mkdir(exist_ok=True, parents=True)
    aparc_aseg = Path(json_dict["T1w_dir"])/"aparc+aseg.nii.gz"
    wmmask_img = generate_tissue_mask(aparc_aseg, "wm")
    wmmask_name = t1reg_dir/"wmmask.nii.gz"
    nb.save(wmmask_img, wmmask_name)

    # find gradient distortion correction warp, fieldmaps and PA epidc warp
    gdc_name = Path(json_dict['ASL_dir'])/'gradient_unwarp/fullWarp_abs.nii.gz'
    gdc_warp = rt.NonLinearRegistration.from_fnirt(
        str(gdc_name), calib_names[0], calib_names[0],
        intensity_correct=True, constrain_jac=(0.01, 100)
    )
    topup_dir = Path(json_dict["ASL_dir"])/"topup"
    fmap, fmapmag, fmapmagbrain = [topup_dir/f"fmap{ext}.nii.gz" 
                                   for ext in ('', 'mag', 'magbrain')]
    epi_dc_warp = rt.NonLinearRegistration.from_fnirt(coefficients=str(topup_dir/"WarpField_01.nii.gz"),
                                                      src=str(fmap),
                                                      ref=str(fmap),
                                                      intensity_correct=True,
                                                      constrain_jac=(0.01, 100))

    # register fieldmapmag to structural image for use in SE-based later
    fmap_struct_dir = topup_dir/"fmap_struct_reg"
    Path(fmap_struct_dir).mkdir(exist_ok=True)
    fsdir = Path(json_dict["T1w_dir"])/f"{subject_dir.parts[-1]}_V1_MR"
    generate_asl2struct(fmapmag, struct_name, fsdir, fmap_struct_dir)
    bbr_fmap2struct = rt.Registration.from_flirt(str(fmap_struct_dir/"asl2struct.mat"), 
                                                 src=str(fmapmag), 
                                                 ref=str(struct_name)) 

    for calib_name in calib_names:
        # get calib_dir and other info
        calib_path = Path(calib_name)
        calib_dir = calib_path.parent
        calib_name_stem = calib_path.stem.split('.')[0]

        # apply gdc and epidc to the calibration image
        gdc_dc_warp = rt.chain(gdc_warp, epi_dc_warp)
        gdc_dc_calib_img = gdc_dc_warp.apply_to_image(calib_name, calib_name, order=interpolation)
        distcorr_dir = calib_dir/"DistCorr"
        distcorr_dir.mkdir(exist_ok=True)
        gdc_dc_calib_name = distcorr_dir/f"gdc_dc_{calib_name_stem}.nii.gz"
        nb.save(gdc_dc_calib_img, gdc_dc_calib_name)

        # apply mt scaling factors to the gradient distortion-corrected calibration image
        if not nobandingcorr:
            mt_sfs = np.loadtxt(mt_factors)
            assert (len(mt_sfs) == gdc_dc_calib_img.shape[2])
            mt_gdc_dc_calib_img = nb.nifti1.Nifti1Image(gdc_dc_calib_img.get_fdata()*mt_sfs, 
                                                    gdc_dc_calib_img.affine)
            mtcorr_dir = calib_dir/"MTCorr"
            mtcorr_dir.mkdir(exist_ok=True)
            mt_gdc_dc_calib_name = mtcorr_dir/f"mtcorr_gdc_dc_{calib_name_stem}.nii.gz"
            nb.save(mt_gdc_dc_calib_img, mt_gdc_dc_calib_name)
            calib_corr_name = mt_gdc_dc_calib_name
        else:
            calib_corr_name = gdc_dc_calib_name

        # get registration to structural
        generate_asl2struct(calib_corr_name, struct_name, fsdir, distcorr_dir)
        asl2struct_name = distcorr_dir/"asl2struct.mat"
        asl2struct_reg = rt.Registration.from_flirt(src2ref=str(distcorr_dir/"asl2struct.mat"),
                                                    src=str(calib_corr_name),
                                                    ref=str(struct_name))
        # invert for struct2calib registration
        struct2calib_reg = asl2struct_reg.inverse()
        struct2calib_name = distcorr_dir/"struct2asl.mat"
        np.savetxt(struct2calib_name, struct2calib_reg.to_flirt(str(struct_name), str(calib_corr_name)))
        
        # register fmapmag to calibration image space
        fmap2calib_reg = rt.chain(bbr_fmap2struct, struct2calib_reg)
        fmapmag_calibspc = fmap2calib_reg.apply_to_image(str(fmapmag),
                                                         str(calib_name),
                                                         order=interpolation)
        biascorr_dir = calib_dir/"BiasCorr"
        sebased_dir = biascorr_dir/"SEbased"
        sebased_dir.mkdir(parents=True, exist_ok=True)
        fmapmag_cspc_name = sebased_dir/f"fmapmag_{calib_name_stem}spc.nii.gz"
        nb.save(fmapmag_calibspc, fmapmag_cspc_name)

        # get brain mask in calibration image space
        fs_brainmask = Path(json_dict["T1w_dir"])/"brainmask_fs.nii.gz"
        aslfs_mask_name = calib_dir/"aslfs_mask.nii.gz"
        aslfs_mask = struct2calib_reg.apply_to_image(src=str(fs_brainmask), 
                                                     ref=str(calib_name),
                                                     order=0)
        aslfs_mask = nb.nifti1.Nifti1Image(np.where(aslfs_mask.get_fdata()>0., 1., 0.),
                                           affine=gdc_dc_calib_img.affine)
        nb.save(aslfs_mask, aslfs_mask_name)

        # get sebased bias estimate
        sebased_cmd = [
            "get_sebased_bias",
            "-i", gdc_dc_calib_name, "-f", fmapmag_cspc_name,
            "-m", aslfs_mask_name, "-o", sebased_dir, 
            "--ribbon", ribbon, "--wmparc", wmparc,
            "--corticallut", corticallut, "--subcorticallut", subcorticallut,
            "--struct2calib", struct2calib_name, "--structural", struct_name, 
            "--debug"
        ]
        subprocess.run(sebased_cmd, check=True)

        # apply dilall to bias estimate
        bias_name = sebased_dir/"sebased_bias_dil.nii.gz"
        dilall_name = biascorr_dir/f"{calib_name_stem}_bias.nii.gz"
        dilall_cmd = ["fslmaths", bias_name, "-dilall", dilall_name]
        subprocess.run(dilall_cmd, check=True)

        # bias correct and mt correct the gdc_dc_calib image
        bias_img = nb.load(dilall_name)
        bc_calib = nb.nifti1.Nifti1Image(gdc_dc_calib_img.get_fdata() / bias_img.get_fdata(),
                                         gdc_dc_calib_img.affine)
        biascorr_name = biascorr_dir / f'{calib_name_stem}_restore.nii.gz'
        nb.save(bc_calib, biascorr_name)

        if not nobandingcorr:
            mt_bc_calib = nb.nifti1.Nifti1Image(bc_calib.get_fdata()*mt_sfs,
                                                bc_calib.affine)
            mtcorr_name = mtcorr_dir / f'{calib_name_stem}_mtcorr.nii.gz'
            nb.save(mt_bc_calib, mtcorr_name)
            calib_corr_name = mtcorr_name
        else:
            calib_corr_name = biascorr_name
        
        # add locations of above files to the json
        important_names = {
            f'{calib_name_stem}_bias' : str(dilall_name),
            f'{calib_name_stem}_corr' : str(calib_corr_name)
        }
        update_json(important_names, json_dict)