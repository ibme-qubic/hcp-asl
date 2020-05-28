from .m0_mt_correction import load_json, update_json
from .initial_bookkeeping import create_dirs
from pathlib import Path
import subprocess
import numpy as np

def run_oxford_asl(subject_dir, nobc=False):
    # load subject's json
    json_dict = load_json(subject_dir)

    # extension for pipeline without banding correction
    if nobc:
        ext = "_nobc"
    else:
        ext = None

    # directory for oxford_asl results
    structasl_dir = Path(json_dict['structasl'])
    oxford_dir = structasl_dir / f'TIs{ext}/OxfordASL'
    pvgm_name = structasl_dir / f'PVEs{ext}/pve_GM.nii.gz'
    pvwm_name = structasl_dir / f'PVEs{ext}/pve_WM.nii.gz'
    calib_name = structasl_dir / f'Calib/Calib0/DistCorr{ext}/calib0_dcorr.nii.gz'
    brain_mask = structasl_dir / f'reg{ext}/ASL_grid_T1w_acpc_dc_restore_brain_mask.nii.gz'
    cmd = [
        "oxford_asl",
        f"-i {json_dict['beta_perf']}",
        f"-o {str(oxford_dir)}",
        "--casl",
        "--ibf=tis",
        "--iaf=diff",
        "--tis=1.7,2.2,2.7,3.2,3.7",
        "--rpts=6,6,6,10,15",
        "--fixbolus",
        "--bolus=1.5",
        "--pvcorr",
        f"-c {str(calib_name)}",
        "--cmethod=single",
        f"-m {str(brain_mask)}",
        f"--pvgm={str(pvgm_name)}",
        f"--pvwm={str(pvwm_name)}",
        "--te=19",
        "--debug",
        "--spatial=off",
        "--slicedt=0.059",
        f"-s {json_dict['T1w_acpc']}",
        f"--sbrain={json_dict['T1w_acpc_brain']}",
        "--sliceband=10"
    ]
    print(" ".join(cmd))
    subprocess.run(" ".join(cmd), shell=True)

    # add oxford_asl directory to the json
    important_names = {
        "oxford_asl": str(oxford_dir)
    }
    update_json(important_names, json_dict)