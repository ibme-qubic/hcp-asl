"""
Functions to perform the tag-control differencing of multi-band 
ASL data which has been rescaled and motion corrected.

The rescaling is accounted for by using a GLM to perform the 
tag-control subtraction, similar to the method described in 
'A framework for motion correction of background suppressed
arterial spin labeling perfusion images acquired with
simultaneous multi‐slice EPI', Y. Suzuki, T.W. Okell, M.A. 
Chappell, M.J.P. van Osch
"""

from .initial_bookkeeping import create_dirs
from .m0_mt_correction import load_json, update_json
from fsl.data.image import Image
from pathlib import Path
import subprocess
import numpy as np

def tag_control_differencing(subject_dir, nobc=False):
    # load subject's json
    json_dict = load_json(subject_dir)

    # extension for pipeline without banding correction
    if nobc:
        ext = "_nobc"
    else:
        ext = None

    # load motion- and distortion- corrected data, Y_moco
    distcorr_dir = Path(json_dict['structasl']) / f'TIs/DistCorr{ext}'
    Y_moco_name = distcorr_dir / 'tis_distcorr.nii.gz'
    Y_moco = Image(str(Y_moco_name))

    if nobc:
        S_st = np.ones_like(Y_moco.data)
    else:
        # load registered scaling factors, S_st
        sfs_name = distcorr_dir / 'combined_scaling_factors.nii.gz'
        S_st = Image(str(sfs_name)).data

    # calculate X_perf = X_tc * S_st
    X_tc = np.ones((1, 1, 1, 86)) * 0.5
    X_tc[0, 0, 0, 0::2] =  -0.5
    X_perf = X_tc * S_st

    # split X_perf and Y_moco into even and odd indices
    X_odd = X_perf[:, :, :, 1::2]
    X_even = X_perf[:, :, :, 0::2]
    Y_odd = Y_moco.data[:, :, :, 1::2]
    Y_even = Y_moco.data[:, :, :, 0::2]

    # calculate B_perf and B_baseline
    B_perf = (Y_odd - Y_even) / (X_odd - X_even)
    B_baseline = (X_odd*Y_even - X_even*Y_odd) / (X_odd - X_even)

    # save both images
    beta_dir_name = Path(json_dict['structasl']) / f'TIs{ext}/Betas'
    create_dirs([beta_dir_name, ])
    B_perf_name = beta_dir_name / 'beta_perf.nii.gz'
    B_perf_img = Image(B_perf, header=Y_moco.header)
    B_perf_img.save(B_perf_name)
    B_baseline_name = beta_dir_name / 'beta_baseline.nii.gz'
    B_baseline_img = Image(B_baseline, header=Y_moco.header)
    B_baseline_img.save(B_baseline_name)

    # add B_perf_name to the json as will be needed in oxford_asl
    important_names = {
        'beta_perf': str(B_perf_name)
    }
    update_json(important_names, json_dict)