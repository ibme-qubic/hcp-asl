import argparse
from pathlib import Path
from hcpasl.initial_bookkeeping import create_dirs
from hcpasl.m0_mt_correction import load_json, update_json
import nibabel as nb
import numpy as np
import subprocess
import scipy as sp

REPEATS = (6, 6, 6, 10, 15)
TIS = (1.7, 2.2, 2.7, 3.2, 3.7)
SIDES = ('L', 'R')

def strip_suffixes(path):
    name = path.name
    name = name.split('.')[0]
    return name

def load_functional(filename):
    # filename is the location of a *.func.gii file
    func = nb.load(filename)
    func_data = np.array([darray.data for darray in func.darrays]).T
    func_meta = func.meta
    return func_data, func_meta

def load_surface(filename):
    # filename is the location of a *.surf.gii file
    gii = nb.load(filename)
    vertices = gii.darrays[0].data # co-ordinates of the vertices
    trigs = gii.darrays[1].data    # indices of vertices in each triangle
    return vertices, trigs, gii

def write_func_data(fname, vertex_data, surf_gii, meta):
    vertex_data = vertex_data.astype(np.float32)
    array = nb.gifti.GiftiDataArray(
        vertex_data, 
        coordsys=None, 
        intent='NIFTI_INTENT_TIME_SERIES', 
        datatype=nb.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'], 
        encoding='GIFTI_ENCODING_ASCII'
    )
    array.coordsys = None
    func_gii = nb.gifti.GiftiImage(surf_gii.header, darrays=[array], meta=meta)
    func_gii.to_filename(fname)
    
def average_tis(fname, sname, rpts, oname):
    """
    Loads functional data which is multi-TI. Calculates the mean 
    functional values at each TI then saves this as oname. Suggested 
    that mean image be saved as {fname}_mean.func.gii.
    ----------
    Parameters
    ----------
    fname : filename of functional data file (STR)
    sname : filename of surface the metric is defined upon (STR)
    rpts  : number of repeats at each TI (inversion times) (list of INTs)
    oname : savename for the mean functional data (STR)
    """
    
    # load files
    func_data, func_meta = load_functional(fname)
    surf_gii = nb.load(sname)

    # calculate average functional images
    cum_rpts = np.cumsum(rpts)
    mean_func = np.array(
        [func_data[:, x:y].mean(axis=1) for x,y in zip(cum_rpts-rpts, cum_rpts)]
    ).T
    # save image
    write_func_data(oname, mean_func, surf_gii, func_meta)

def project_for_fabber():
    pass

def do_basil(perf_name, mean_name, surf_name, tiimg_name, out_dir):
    # load surface
    repeats = REPEATS
    base_command = [
        "fabber_asl",
        f"--surface={surf_name}",
        "--model=aslrest",
        "--casl",
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
        f"--tiimg={tiimg_name}"
    ]
    # iteration-specific options
    for iteration in range(5):
        it_command = base_command.copy()
        # data
        if iteration == 0 or iteration == 1:
            data = mean_name
        else:
            data = perf_name
        it_command.append(f"--data={data}")
        # inferart
        if iteration==1 or iteration==3:
            it_command.append("--inferart")
        # output dir
        temp_out = out_dir / f'fabber_{iteration}'
        it_command.append(f"--output={temp_out}")
        # continue from mvn
        if iteration != 0:
            mvn = temp_out.parent / f'fabber_{iteration - 1}/finalMVN.nii.gz'
            it_command.append(f"--continue-from-mvn={mvn}")
        # repeats
        if iteration >= 2:
            for n, repeat in enumerate(repeats):
                it_command.append(f"--rpt{n+1}={repeat}")
        # spatial or not
        if iteration < 4:
            method = "vb"
        else:
            method = "spatialvb"
        it_command.append(f"--method={method}")
        # run
        print(" ".join(it_command))
        subprocess.run(it_command, check=True)

        # threshold from below at 0
        ftiss_name = temp_out / 'mean_ftiss.func.gii'
        cmd = [
            "wb_command",
            "-metric-math",
            "max(x, 0)",
            str(ftiss_name),
            "-var", "x", f"{str(ftiss_name)}"
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

def surface_analysis(subject_dir):
    # debugging
    force_refresh = True

    # argument handling
    subject_dir = Path(subject_dir)

    # load json
    json_dict = load_json(subject_dir)

    # project the beta_perf series
    beta_perf = Path(subject_dir) / 'T1w/ASL/TIs/Betas/beta_perf.nii.gz'
    projected_beta_dir = beta_perf.parent / 'projected_betas'
    stripped_beta_perf = strip_suffixes(beta_perf)
    create_dirs([projected_beta_dir, ])
    for side in SIDES:
        # surface names
        mid_name = json_dict[f'{side}_mid']
        pial_name = json_dict[f'{side}_pial']
        white_name = json_dict[f'{side}_white']

        # projected beta_perf name
        proj_perf_name = projected_beta_dir/f'{side}_{stripped_beta_perf}.func.gii'
        if not proj_perf_name.exists() or force_refresh:
            cmd = [
                "wb_command",
                "-volume-to-surface-mapping",
                beta_perf,
                mid_name,
                proj_perf_name,
                "-ribbon-constrained",
                white_name,
                pial_name
            ]
            print(f'Running command: {cmd}')
            subprocess.run(cmd)
    
    # obtain surface mean
    for side in SIDES:
        perf_name = projected_beta_dir/f'{side}_{stripped_beta_perf}.func.gii'
        mean_name = projected_beta_dir/f'{side}_{stripped_beta_perf}_mean.func.gii'
        surf_name = json_dict[f'{side}_mid']
        if not mean_name.exists() or force_refresh:
            average_tis(str(perf_name), str(surf_name), REPEATS, str(mean_name))
    
    # reformat projected results to be in the form that fabber expects
    for side in SIDES:
        perf_name = projected_beta_dir/f'{side}_{stripped_beta_perf}.func.gii'
        surf_name = json_dict[f'{side}_mid']
        perf_data, perf_meta = load_functional(str(perf_name))
        surf_gii = nb.load(surf_name)
        write_func_data(str(perf_name), perf_data, surf_gii, perf_meta)

    # project the timing image
    tiimg = subject_dir / 'T1w/ASL/timing_img.nii.gz'
    for side in SIDES:
        # surface names
        mid_name = json_dict[f'{side}_mid']
        pial_name = json_dict[f'{side}_pial']
        white_name = json_dict[f'{side}_white']

        # projected beta_perf name
        proj_tiimg_name = tiimg.parent / f'{side}_timing.func.gii'
        if not proj_tiimg_name.exists() or force_refresh:
            cmd = [
                "wb_command",
                "-volume-to-surface-mapping",
                tiimg,
                mid_name,
                proj_tiimg_name,
                "-enclosing"
            ]
            print(f'Running command: {cmd}')
            subprocess.run(cmd)

    # reformat projected tiimg to be in the form that fabber expects
    for side in SIDES:
        tiimg_name = tiimg.parent / f'{side}_timing.func.gii'
        surf_name = json_dict[f'{side}_mid']
        tiimg_data, tiimg_meta = load_functional(str(tiimg_name))
        surf_gii = nb.load(surf_name)
        write_func_data(str(tiimg_name), tiimg_data, surf_gii, tiimg_meta)

    # run fabber
    for side in SIDES:
        perf_name = projected_beta_dir/f'{side}_{stripped_beta_perf}.func.gii'
        mean_name = projected_beta_dir/f'{side}_{stripped_beta_perf}_mean.func.gii'
        surf_name = json_dict[f'{side}_mid']
        tiimg_name = tiimg.parent / f'{side}_timing.func.gii'
        out_dir = projected_beta_dir / f'{side}_basil'
        create_dirs([out_dir, ])
        do_basil(perf_name, mean_name, surf_name, tiimg_name, out_dir)