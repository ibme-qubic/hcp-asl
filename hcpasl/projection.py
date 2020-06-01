from .initial_bookkeeping import create_dirs
from .m0_mt_correction import load_json, update_json
from pathlib import Path
import subprocess
from fsl.wrappers.flirt import applyxfm
from itertools import product

def project_to_surface(subject_dir):
    # load subject's json
    json_dict = load_json(subject_dir)

    # perfusion calib and variance calib
    pc_name = Path(json_dict['oxford_asl']) / 'struct_space/perfusion_calib.nii.gz'
    vc_name = Path(json_dict['oxford_asl']) / 'struct_space/perfusion_var_calib.nii.gz'
    names = (pc_name, vc_name)

    # create directory for surface results
    projection_dir = Path(json_dict['oxford_asl']) / 'SurfaceResults32k'
    create_dirs([projection_dir, ])
    sides = ('L', 'R')

    for name, side in product(names, sides):
        # surface file names
        mid_name = json_dict[f'{side}_mid']
        pial_name = json_dict[f'{side}_pial']
        white_name = json_dict[f'{side}_white']

        # get stem name
        stem = name.stem.strip('.nii')

        # save name
        savename = projection_dir / f'{side}_{stem}.func.gii'
        cmd = [
            "wb_command",
            "-volume-to-surface-mapping",
            pc_name,
            mid_name,
            savename,
            "-ribbon-constrained",
            white_name,
            pial_name
        ]
        subprocess.run(cmd)