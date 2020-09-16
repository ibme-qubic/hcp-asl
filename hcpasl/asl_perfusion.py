from .m0_mt_correction import load_json, update_json
from .initial_bookkeeping import create_dirs
from pathlib import Path
import subprocess
import numpy as np
from fsl.data.image import Image
from fsl.wrappers import fslmaths

def run_fabber_asl(subject_dir, target='structural'):
    json_dict = load_json(subject_dir)
    structasl_dir = Path(json_dict['structasl'])
    brain_mask = structasl_dir / 'reg/ASL_grid_T1w_acpc_dc_restore_brain_mask.nii.gz'
    timing_image = structasl_dir / 'timing_img.nii.gz'

    # take mean of data for first 2 iterations
    beta_perf_name = structasl_dir / 'TIs/Betas/beta_perf.nii.gz'
    beta_perf_mean_name = structasl_dir / 'TIs/Betas/beta_perf_mean.nii.gz'
    repeats = [6, 6, 6, 10, 15]
    mean_call = [
        "asl_file",
        f"--data={beta_perf_name}",
        "--ntis=5",
        "--ibf=tis",
        "--rpts=6,6,6,10,15",
        "--iaf=diff",
        "--obf=tis",
        f"--mean={beta_perf_mean_name}"
    ]
    subprocess.run(mean_call, check=True)

    base_command = [
        "fabber_asl",
        "--model=aslrest",
        "--method=vb",
        "--casl",
        f"--mask={brain_mask}",
        "--save-mvn",
        "--overwrite",
        "--incart",
        "--inctiss",
        "--infertiss",
        "--incbat",
        "--inferbat",
        "--noise=white",
        "--save-mean",
        "--tau=1.5",
        "--bat=1.3",
        "--batsd=1.0",
        "--allow-bad-voxels",
        "--convergence=trialmode",
        "--data-order=singlefile",
        "--disp=none",
        "--exch=mix",
        "--max-iterations=20",
        "--max-trials=10",
        f"--tiimg={timing_image}"
    ]
    # iteration-specific options
    for iteration in range(4):
        it_command = base_command.copy()
        # data
        if iteration == 0 or iteration == 1:
            data = beta_perf_mean_name
        else:
            data = beta_perf_name
        it_command.append(f"--data={data}")
        # inferart
        if iteration==1 or iteration==3:
            it_command.append("--inferart")
        # output dir
        out_dir = beta_perf_name.parent / f'fabber_{iteration}'
        it_command.append(f"--output={out_dir}")
        # continue from mvn
        if iteration != 0:
            mvn = beta_perf_name.parent / f'fabber_{iteration - 1}/finalMVN.nii.gz'
            it_command.append(f"--continue-from-mvn={mvn}")
        # repeats
        if iteration >= 2:
            for n, repeat in enumerate(repeats):
                it_command.append(f"--rpt{n+1}={repeat}")
        # run
        subprocess.run(it_command, check=True)
    # threshold last run's perfusion estimate
    mean_ftiss = str(out_dir / 'mean_ftiss.nii.gz')
    fslmaths(mean_ftiss).thr(0).run(mean_ftiss)
    # add oxford_asl directory to the json
    important_names = {
        "oxford_asl": str(out_dir)
    }
    update_json(important_names, json_dict)

def run_oxford_asl(subject_dir, target='structural'):
    """
    Run oxford_asl on the HCP's ASL data.
    """
    # load subject's json
    json_dict = load_json(subject_dir)

    # base oxford_asl options common to both cases
    cmd = [
        "oxford_asl",
        f"-i {json_dict['beta_perf']}",
        "--casl",
        "--ibf=tis",
        "--iaf=diff",
        "--rpts=6,6,6,10,15",
        "--fixbolus",
        "--bolus=1.5",
        "--te=19",
        "--debug",
        "--spatial=off"
    ]

    # additional run-specific options
    if target == 'asl':
        oxford_dir = Path(json_dict['TIs_dir']) / 'OxfordASL'
        brain_mask = Path(json_dict['structasl']) / 'reg/asl_vol1_mask_init.nii.gz'
        extra_args = [
            f"-o {str(oxford_dir)}",
            f"-m {str(brain_mask)}",
            "--tis=1.7,2.2,2.7,3.2,3.7",
            "--slicedt=0.059",
            "--sliceband=10"
        ]
    else:
        structasl_dir = Path(json_dict['structasl'])
        oxford_dir = structasl_dir / 'TIs/OxfordASL'
        pvgm_name = structasl_dir / 'PVEs/pve_GM.nii.gz'
        pvwm_name = structasl_dir / 'PVEs/pve_WM.nii.gz'
        csf_mask_name = structasl_dir / 'PVEs/vent_csf_mask.nii.gz'
        calib_name = structasl_dir / 'Calib/Calib0/DistCorr/calib0_dcorr.nii.gz'
        brain_mask = structasl_dir / 'reg/ASL_grid_T1w_acpc_dc_restore_brain_mask.nii.gz'
        timing_image = structasl_dir / 'timing_img.nii.gz'
        extra_args = [
            f"-o {str(oxford_dir)}",
            f"--pvgm={str(pvgm_name)}",
            f"--pvwm={str(pvwm_name)}",
            f"--csf={str(csf_mask_name)}",
            f"-c {str(calib_name)}",
            f"-m {str(brain_mask)}",
            f"--tiimg={timing_image}"
            ]
    cmd = cmd + extra_args
    print(" ".join(cmd))
    subprocess.run(" ".join(cmd), shell=True)

    # add oxford_asl directory to the json
    important_names = {
        "oxford_asl": str(oxford_dir)
    }
    update_json(important_names, json_dict)