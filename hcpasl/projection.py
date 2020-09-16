from .initial_bookkeeping import create_dirs
from .m0_mt_correction import load_json, update_json
from pathlib import Path
import subprocess
from fsl.wrappers.flirt import applyxfm
from itertools import product
import regtricks as rt
import multiprocessing as mp
import nibabel as nb

def project_to_surface(subject_dir, target='structural'):
    """
    Project the results of the pipeline to the cortical surface.
    """
    # load subject's json
    json_dict = load_json(subject_dir)

    # perfusion calib and variance calib
    oxasl_dir = Path(json_dict['oxford_asl'])
    pc_name = oxasl_dir / 'native_space/perfusion.nii.gz'
    vc_name = oxasl_dir / 'native_space/perfusion_var.nii.gz'

    # if in ASL space, need to register to T1w
    if target == 'asl':
        ref = subject_dir/"T1w/ASL/reg/ASL_grid_T1w_acpc_dc_restore.nii.gz"
        asl_t1_name = subject_dir/"T1w/T1w_acpc_dc_restore.nii.gz"
        asl2struct = rt.Registration.from_flirt(
            str(oxasl_dir.parent/"DistCorr/asl2struct.mat"),
            src=str(pc_name),
            ref=str(asl_t1_name)
        )
        t1_pc = asl2struct.apply_to_image(
            src=str(pc_name),
            ref=str(ref),
            order=3,
            cores=mp.cpu_count()
        )
        pc_name = pc_name.parent/"asl_t1_perfusion.nii.gz"
        nb.save(t1_pc, str(pc_name))
        t1_vc = asl2struct.apply_to_image(
            src=str(vc_name),
            ref=str(ref),
            order=3,
            cores=mp.cpu_count()
        )
        vc_name = vc_name.parent/"asl_t1_perfusion_var.nii.gz"
        nb.save(t1_vc, str(vc_name))

    names = (pc_name, vc_name)

    # create directory for surface results
    projection_dir = oxasl_dir / 'SurfaceResults32k'
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
            name,
            mid_name,
            savename,
            "-ribbon-constrained",
            white_name,
            pial_name
        ]
        subprocess.run(cmd)