"""
This contains a range of functions required to correct for the 
Magnetisation Transfer effect visible in the HCP data.
"""

import json
from pathlib import Path
from fsl.wrappers import fslmaths, LOAD, bet, fast
from .initial_bookkeeping import create_dirs
import subprocess

def load_json(subject_dir):
    """
    Load json but with some error-checking to make sure it exists.
    If it doesn't exist, instruct user to run the first part of 
    the pipeline first.
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
    Adds the key-value pairs in `new_dict` to the `old_dict` 
    and save the resulting dictionary to the json found in 
    `old_dict['json_name']`.

    Inputs:
        - `new_dict` = dictionary containing new key-value 
            pairs to add to the json of important file 
            names
        - `old_dict` = dictionary to be updated, also 
            has a field containing the location of the json.
    """
    old_dict.update(new_dict)
    with open(Path(old_dict['json_name']), 'w') as fp:
        json.dump(old_dict, fp, sort_keys=True, indent=4)

def correct_M0(subject_dir, mt_factors=None):
    """
    Correct the M0 images for a particular subject whose data 
    is stored in `subject_dir`. The corrections to be 
    performed include:
        - Bias-field correction
        - Magnetisation Transfer correction
    
    Inputs
        - `subject_dir` = pathlib.Path object specifying the 
            subject's base directory
        - `mt_factors` = pathlib.Path object specifying the 
            location of empirically estimated MT correction 
            scaling factors
    """
    # load json containing info on where files are stored
    json_dict = load_json(subject_dir)
    
    # do for both m0 images for the subject, calib0 and calib1
    calib_names = [json_dict['calib0_img'], json_dict['calib1_img']]
    for calib_name in calib_names:
        # get calib_dir and other info
        calib_path = Path(calib_name)
        calib_dir = calib_path.parent
        calib_name_stem = calib_path.stem.split('.')[0]

        # run BET on m0 image
        betted_m0 = bet(calib_name, LOAD)

        # create directories to store results
        fast_dir = calib_dir / 'FAST'
        biascorr_dir = calib_dir / 'BiasCorr'
        mtcorr_dir = calib_dir / 'MTCorr'
        create_dirs([fast_dir, biascorr_dir, mtcorr_dir])

        # estimate bias field on brain-extracted m0 image
            # run FAST, storing results in directory
        fast_base = fast_dir / calib_name_stem
        fast(
            betted_m0['output'], # output of bet
            out=str(fast_base), 
            type=3, # image type, 3=PD image
            b=True, # output estimated bias field
            nopve=True # don't need pv estimates
        )
        bias_name = fast_dir / f'{calib_name_stem}_bias.nii.gz'

        # apply bias field to original m0 image (i.e. not BETted)
        biascorr_name = biascorr_dir / f'{calib_name_stem}_restore.nii.gz'
        fslmaths(calib_name).div(str(bias_name)).run(str(biascorr_name))
        important_names = {
            f'{calib_name_stem}_bias' : str(bias_name),
            f'{calib_name_stem}_bc' : str(biascorr_name)
        }

        # apply mt_factors to bias-corrected m0 image if given
        if mt_factors:
            mtcorr_name = mtcorr_dir / f'{calib_name_stem}_mtcorr.nii.gz'
            fslmaths(str(biascorr_name)).mul(str(mt_factors)).run(str(mtcorr_name))
            important_names.update({f'{calib_name_stem}_mc' : str(mtcorr_name)})

        # add locations of above files to the json
        update_json(important_names, json_dict)