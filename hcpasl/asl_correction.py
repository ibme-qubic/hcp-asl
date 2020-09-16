"""
Functions for the correction of HCP ASL data.

Corrections to be applied include:
    - Bias-field correction (using the bias field estimated on the 
        subject's calibration image - currently the first 
        calibration image but maybe this should be changed to the 
        mean of the first and the second? and maybe there should be 
        some registration from the calibration image to the mean of 
        the ASL series to register the bias field to the series?)
    - MT correction (correction of the Magnetisation Transfer effect 
        visible in HCP ASL data using empirically estimated scaling 
        coefficients)
    - Saturation recovery (post-MT correction, the saturation 
        recovery expected in ASL is much more visible, giving a 
        residual banding effect)
    - Initial slice-timing correction (done using the parameter 
        estimates obtained above. Initial because the satrecov 
        model is initially estimated upon data which hasn't been 
        motion corrected - we account for this later)
    - Motion estimation
    - Second slice-timing correction (use the motion estimates 
        from above to align the parameter estimates with individual 
        frames of the ASL series prior to correction. Perform 
        slice-timing correction again using these aligned 
        parameter maps.)
    - Registration (apply motion estimates to the final slice-timing 
        corrected ASL series from the above step to obtain final, 
        fully-corrected, registered ASL series)
"""

from .initial_bookkeeping import create_dirs
from .m0_mt_correction import load_json, update_json
from fsl.wrappers import fslmaths, LOAD
from fsl.wrappers.flirt import mcflirt, applyxfm, applyxfm4D
from fsl.data.image import Image
import nibabel as nib
from fabber import Fabber, percent_progress
import sys
from pathlib import Path
import shutil
import subprocess
import numpy as np
import regtricks as rt
import multiprocessing as mp
def _satrecov_worker(control_name, satrecov_dir, tis, rpts, ibf, spatial):
    """
    Wrapper for fabber's saturation recovery model.

    Parameters
    ----------
    control_name : pathlib.Path
        Path to the control images.
    satrecov_dir : pathlib.Path
        Parent directory for the satrecov results. Results from 
        this will be stored either in {`satrecov_dir`}/spatial 
        or {`satrecov_dir`}/nospatial depending on value of 
        'spatial`.
    tis : list
        TIs for the ASL sequence.
    rpts : list
        Number of repeats for each TI in the sequence.
    ibf : str
        Input format of the sequence.
    spatial : bool
        Choose whether to run fabber in spatial (True) or 
        non-spatial (False) mode.
    """
    # set options for Fabber run, generic to spatial and non-spatial runs
    options = {
        'data': str(control_name),
        'overwrite': True,
        'noise': 'white',
        'ibf': ibf,
        'model': 'satrecov',
        'save-mean': True, # is this needed in nonspatial run?
        'casl': True,
        'slicedt': 0.059,
        'ti1': tis[0],
        'ti2': tis[1],
        'ti3': tis[2],
        'ti4': tis[3],
        'ti5': tis[4],
        'sliceband': 10,
        'bolus': 1.5,
        'rpt1': rpts[0],
        'rpt2': rpts[1],
        'rpt3': rpts[2],
        'rpt4': rpts[3],
        'rpt5': rpts[4],
        'fixa': True
    }
    # spatial or non-spatial specific options
    spatial_dir = satrecov_dir / 'spatial'
    nospatial_dir = satrecov_dir / 'nospatial'
    if spatial:
        out_dir = str(spatial_dir)
        extra_options = {
            'method': 'spatialvb',
            'output': out_dir,
            'continue-from-mvn': str(nospatial_dir / 'finalMVN.nii.gz')
        }
    else:
        out_dir = str(nospatial_dir)
        extra_options = {
            'method': 'vb',
            'output': out_dir,
            'save-mvn': True
        }
    options.update(extra_options)
    # run Fabber
    fab = Fabber()
    run = fab.run(options, progress_cb=percent_progress(sys.stdout))# Basic interaction with the run output
    # info about fabber run
    print("\nOutput data summary")
    for name, data in run.data.items():
        print("%s: %s" % (name, data.shape))
    print("Run finished at: %s" % run.timestamp_str)
    # Write full contents out to a directory
    # load control image to get header for saving
    control_img = Image(str(control_name))
    run.write_to_dir(out_dir, ref_nii=control_img)

def _split_tag_control(asl_name, ntis, iaf, ibf, rpts):
    """
    Split an ASL sequence into its tag and control images.

    Parameters
    ----------
    asl_name : pathlib.Path
        Path to the ASL series to be split.
    ntis : int
        Number of TIs in the ASL sequence.
    iaf : str
        ("tc", "ct", "diff")
    ibf : str
        ("tis", "rpts")
    rpts : list
        List of number of repeats at each TI.
    
    Returns
    -------
    even_name : pathlib.Path
    odd_name : pathlib.Path
    """
    asl_base = asl_name.parent / asl_name.stem.split('.')[0]
    # messy - is there a better way of filling in arguments?
    cmd = [
        'asl_file',
        f'--data={str(asl_name)}',
        f'--ntis={ntis}',
        f'--iaf={iaf}',
        f'--ibf={ibf}',
        '--spairs',
        f'--rpts={rpts[0]},{rpts[1]},{rpts[2]},{rpts[3]},{rpts[4]}',
        f'--out={asl_base}'
    ]
    subprocess.run(cmd)
    even_name = asl_name.parent / f'{asl_base}_even.nii.gz'
    odd_name = asl_name.parent / f'{asl_base}_odd.nii.gz'
    return even_name, odd_name

def _saturation_recovery(asl_name, results_dir, ntis, iaf, ibf, tis, rpts):
    """
    Use Fabber's `satrecov` model to estimate a T1 map.

    Split the ASL sequence into tag and control images. Fit the 
    `satrecov` model on the control images, first with Fabber's 
    spatial mode off, then with it on.
    
    Parameters
    ----------
    asl_name : pathlib.Path
        Path for the ASL series on which the model will be 
        estimated.
    results_dir : pathlib.Path
        Directory in which to save the `nospatial` and `spatial` 
        parameter estimates.
    """
    # obtain control images of ASL series
    control_name, tag_name = _split_tag_control(asl_name, ntis, iaf, ibf, rpts)
    # satrecov nospatial
    _satrecov_worker(control_name, results_dir, tis, rpts, ibf, spatial=False)
    # satrecov spatial
    _satrecov_worker(control_name, results_dir, tis, rpts, ibf, spatial=True)
    t1_name = results_dir / 'spatial/mean_T1t.nii.gz'
    return t1_name

def _fslmaths_med_filter_wrapper(image_name):
    """
    Simple wrapper for fslmaths' median filter function. Applies 
    the median filter to `image_name`. Derives and returns the 
    name of the filtered image as {image_name}_filt.nii.gz.
    """
    filtered_name = image_name.parent / f'{image_name.stem.split(".")[0]}_filt.nii.gz'
    cmd = [
        'fslmaths',
        image_name,
        '-fmedian',
        filtered_name
    ]
    subprocess.run(cmd)
    return filtered_name

def _slicetiming_correction(
    asl_name, t1_name, tis, rpts, 
    slicedt, sliceband, n_slices
    ):
    """
    Correct an ASL sequence for the slice-timing effect.

    satrecov_model: S(t) = M_0 * (1 - exp{-t/T1t})

    Divides the value in a voxel by the above model evaluated at 
    t = TI + n*slicedt where n is the slice number of the voxel 
    within a band. This accounts for the voxel being imaged at a 
    different time than its supposed TI.

    Multiplies the value in a voxel by the above model evaluated 
    at t = TI, i.e. scales the values as if they had been imaged 
    at the TI that was specified in the ASL sequence.

    Parameters
    ----------
    asl_name : pathlib.Path
        Path to the ASL sequence.
    t1_name : pathlib.Path
        Path to the T1 estimate. This T1 estimate can be a single 
        volume or a time-series. If it is a time-series, the 
        number of time points must match that of the ASL series.
    tis : list
        List of TIs used of the sequence.
    rpts : list
        List of repeats. This is the number of repeats for each 
        TI.
    slicedt : float
        Time to acquire a slice in the sequence.
    sliceband : int
        Number of slices in each band in the multi-band 
        acquisition.
    n_slices : int
        Number of slices in the sequence.

    Returns
    -------
    stcorr_img : fsl.image.Image
        Slice-time corrected ASL sequence.
    stcorr_factors_img : fsl.image.Image
        Scaling factors used to perform the slice-time 
        correction.
    """
    # timing information for scan
    # supposed measurement time of slice
    tis_array = np.repeat(np.array(tis), 2*np.array(rpts)).reshape(1, 1, 1, -1)
    # actual measurment time of slice
    slice_numbers = np.tile(
        np.arange(0, sliceband),
        n_slices // sliceband
    ).reshape(1, 1, -1, 1)
    slice_times = tis_array + (slicedt * slice_numbers)
    # load images
    asl_img = Image(str(asl_name))
    t1_img = Image(str(t1_name))
    # check dimensions of t1 image to see if time series or not
    if t1_img.ndim == 3:
        t1_data = t1_img.data[..., np.newaxis]
    elif t1_img.ndim == 4:
        t1_data = t1_img.data
    # multiply asl sequence by satrecov model evaluated at TI
    numexp = - tis_array / t1_data
    num = 1 - np.exp(numexp)
    # divide asl sequence by satrecov model evaluated at actual slice-time
    denexp = - slice_times / t1_data
    den = 1 - np.exp(denexp)
    # evaluate scaling factors
    stcorr_factors = num / den
    stcorr_factors_img = Image(stcorr_factors, header=asl_img.header)
    # correct asl series
    stcorr_data = asl_img.data * stcorr_factors
    stcorr_img = Image(stcorr_data, header=asl_img.header)
    return stcorr_img, stcorr_factors_img

def _register_param(param_name, transform_dir, reffile, param_reg_name):
    """
    Apply motion estimates to an image to obtain a time series.

    Given a parameter map, `param_name`, and a series of motion 
    estimates, `transform_dir`, apply the motion estimates to 
    the parameter map and obtain a time series of the map 
    in the frame described by the motion estimates.

    Uses fsl's applyxfm.

    Parameters
    ----------
    param_name : pathlib.Path
        Path to the parameter estimate to which we want to apply 
        motion.
    transform_dir : pathlib.Path
        Path to a directory containing mcflirt motion estimates.
    reffile : pathlib.Path
        Path for reference image in mcflirt motion estimate.
    param_reg_name : pathlib.Path
        Savename of output file
    """
    # list of transformations in transform_dir
    transforms = sorted(transform_dir.glob('**/*'))
    out_names = []
    for n, transform in enumerate(transforms):
        # naming intermediate file
        if n < 10:
            out_n = param_reg_name.parent / f'{param_reg_name.stem.split(".")[0]}_000{n}.nii.gz'
        else:
            out_n = param_reg_name.parent / f'{param_reg_name.stem.split(".")[0]}_00{n}.nii.gz'
        out_names.append(out_n)
        applyxfm(str(param_name), str(reffile), str(transform), str(out_n))
        cmd = [
            "applyxfm4D",
            str(param_name),
            str(reffile),
            str(out_n),
            str(transform),
            "-singlematrix"
        ]
        subprocess.run(cmd)
    # merge registered parameter volumes into one time series
    cmd = [
        'fslmerge',
        '-t',
        str(param_reg_name),
        *out_names
    ]
    subprocess.run(cmd)
    # remove intermediate file names
    for out_n in out_names:
        out_n.unlink()

def hcp_asl_moco(subject_dir, mt_factors, superlevel=1, cores=mp.cpu_count(), order=3):
    """
    Full ASL correction and motion estimation pipeline.

    This function performs the full motion-correction pipeline for 
    the HCP ASL data. The steps of the pipeline are:
    #. Bias-field correction using the bias-field estimated from 
    the first calibration image;
    #. MT correction using pre-calculated correction factors;
    #. An initial fit of the `satrecov` model on the bias and 
    MT-corrected series;
    #. Median filtering of the estimated T1 map;
    #. Initial slice-time correction using the median filtered 
    T1 map;
    #. Motion estimation;
    #. Apply the motion estimates to the pre-slicetime-correction 
    ASL series;
    #. Second fit of the 'satrecov' model on motion-corrected 
    series;
    #. Apply motion estimates to resulting T1 map so that it is 
    aligned with each of the volumes in the ASL series;
    #. Refined slice-timing correction

    Parameters
    ----------
    subject_dir : pathlib.Path
        Path to the subject's base directory.
    mt_factors : pathlib.Path
        Path to the pre-calculated MT correction scaling factors.
    superlevel : int, optional
        Superlevel to use when using regtricks. Default is 1.
    cores : int, optional
        Number of cores regtricks will use. Default is the number 
        of cores available.
    order : int, optional
        Order of interpolation to be used by regtricks. This is 
        passed to scipy's map_coordinates. See that for more 
        information. Default is 3.
    """
    # asl sequence parameters
    ntis = 5
    iaf = "tc"
    ibf = "tis"
    tis = [1.7, 2.2, 2.7, 3.2, 3.7]
    rpts = [6, 6, 6, 10, 15]
    slicedt = 0.059
    sliceband = 10
    n_slices = 60
    # load json containing important file info
    json_dict = load_json(subject_dir)
    # original ASL series and bias field names
    asl_name = Path(json_dict['ASL_seq'])
    bias_name = json_dict['calib0_bias']
    # create directories for results
    tis_dir_name = Path(json_dict['TIs_dir'])
    bcorr_dir = tis_dir_name / 'BiasCorr'
    mtcorr_dir = tis_dir_name / 'MTCorr'
    satrecov_dir = tis_dir_name / 'SatRecov'
    stcorr_dir = tis_dir_name / 'STCorr'
    moco_dir = tis_dir_name / 'MoCo'
    asln2m0_name = moco_dir / 'asln2m0.mat'
    m02asln_name = moco_dir / 'm02asln.mat'
    asln2asl0_name = moco_dir / 'asln2asl0.mat'
    asl02asln_name = moco_dir / 'asl02asln.mat'
    create_dirs([
        tis_dir_name,
        bcorr_dir,
        mtcorr_dir,
        satrecov_dir,
        stcorr_dir,
        moco_dir,
        asln2m0_name,
        m02asln_name,
        asln2asl0_name,
        asl02asln_name
    ])
    # bias correct the original ASL series
    bcorr_img = bcorr_dir / 'tis_biascorr.nii.gz'
    fslmaths(str(asl_name)).div(str(bias_name)).run(str(bcorr_img))
    # apply MT scaling factors to the bias-corrected ASL series
    mtcorr_name = mtcorr_dir / 'tis_mtcorr.nii.gz'
    # load mt factors
    mt_sfs = np.loadtxt(mt_factors)
    biascorr_img = Image(str(bcorr_img))
    assert (len(mt_sfs) == biascorr_img.shape[2])
    mtcorr_img = Image(biascorr_img.data*mt_sfs.reshape(1, 1, -1, 1), header=biascorr_img.header)
    mtcorr_img.save(str(mtcorr_name))
    # estimate satrecov model on bias and MT corrected ASL series
    t1_name = _saturation_recovery(mtcorr_name, satrecov_dir, ntis, iaf, ibf, tis, rpts)
    t1_filt_name = _fslmaths_med_filter_wrapper(t1_name)
    # perform slice-time correction using estimated tissue params
    stcorr_img, stfactors_img = _slicetiming_correction(mtcorr_name, t1_filt_name, tis, rpts, slicedt, sliceband, n_slices)
    stcorr_name = stcorr_dir / 'tis_stcorr.nii.gz'
    stcorr_img.save(stcorr_name)
    stfactors_name = stcorr_dir / 'st_scaling_factors.nii.gz'
    stfactors_img.save(stfactors_name)
    # register ASL series to calibration image
    reg_name = moco_dir / 'initial_registration_TIs.nii.gz'
    mcflirt(stcorr_img, reffile=json_dict['calib0_mc'], mats=True, out=str(reg_name))
    # rename mcflirt matrices directory
    orig_mcflirt = moco_dir / 'initial_registration_TIs.nii.gz.mat'
    if asln2m0_name.exists():
        shutil.rmtree(asln2m0_name)
    orig_mcflirt.rename(asln2m0_name)
    # get motion estimates from ASLn to ASL0 (and their inverses)
    asl2m0_list = sorted(asln2m0_name.glob('**/MAT*'))
    m02asl0 = np.linalg.inv(np.loadtxt(asl2m0_list[0]))
    for n, xform in enumerate(asl2m0_list):
        if n == 0:
            fwd_xform = np.eye(4)
        else:
            fwd_xform = m02asl0 @ np.loadtxt(xform)
        inv_xform = np.linalg.inv(fwd_xform)
        np.savetxt(m02asln_name / xform.stem, np.linalg.inv(np.loadtxt(xform)))
        np.savetxt(asln2asl0_name / xform.stem, fwd_xform)
        np.savetxt(asl02asln_name / xform.stem, inv_xform)
    # register pre-ST-correction ASLn to ASL0
    temp_reg_mtcorr = moco_dir / 'temp_reg_tis_mtcorr.nii.gz'
    asln2m0_moco = rt.MotionCorrection.from_mcflirt(
        str(asln2m0_name),
        str(mtcorr_name),
        json_dict['calib0_mc']
    )
    asln2asl0 = rt.chain(asln2m0_moco, asln2m0_moco.transforms[0].inverse())
    reg_mtcorr = Image(asln2asl0.apply_to_image(
        str(mtcorr_name), 
        json_dict['calib0_mc'],
        superlevel=superlevel,
        cores=cores,
        order=order
    ))
    reg_mtcorr.save(str(temp_reg_mtcorr))

    # estimate satrecov model on motion-corrected data
    satrecov_dir = tis_dir_name / 'SatRecov2'
    stcorr_dir = tis_dir_name / 'STCorr2'
    create_dirs([satrecov_dir, stcorr_dir])
    t1_name = _saturation_recovery(temp_reg_mtcorr, satrecov_dir, ntis, iaf, ibf, tis, rpts)
    t1_filt_name = _fslmaths_med_filter_wrapper(t1_name)
    # apply asl0 to asln registrations to new t1 map
    reg_t1_filt_name = t1_filt_name.parent / f'{t1_filt_name.stem.split(".")[0]}_reg.nii.gz'
    reg_t1_filt = Image(
        asln2asl0.inverse().apply_to_image(
            str(t1_filt_name),
            json_dict['calib0_mc'],
            superlevel=superlevel,
            cores=cores,
            order=order
        )
    )
    reg_t1_filt.save(str(reg_t1_filt_name))
    # perform slice-time correction using estimated tissue params
    stcorr_img, stfactors_img = _slicetiming_correction(mtcorr_name, reg_t1_filt_name, tis, rpts, slicedt, sliceband, n_slices)
    # save images
    stcorr_name = stcorr_dir / 'tis_stcorr.nii.gz'
    stfactors_name = stcorr_dir / 'st_scaling_factors.nii.gz'
    stcorr_img.save(str(stcorr_name))
    stfactors_img.save(str(stfactors_name))
    # combined MT and ST scaling factors
    combined_factors_name = stcorr_dir / 'combined_scaling_factors.nii.gz'
    combined_factors_img = Image(stfactors_img.data*mt_sfs.reshape(1, 1, -1, 1), header=stfactors_img.header)
    combined_factors_img.save(str(combined_factors_name))
    # save locations of important files in the json
    important_names = {
        'ASL_stcorr': str(stcorr_name),
        'scaling_factors': str(combined_factors_name)
    }
    update_json(important_names, json_dict)