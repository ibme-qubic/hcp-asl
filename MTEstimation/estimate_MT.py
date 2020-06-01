"""
Set of functions for estimating the MT effect
"""
from hcpasl.m0_mt_correction import load_json, update_json
from hcpasl.initial_bookkeeping import create_dirs
from fsl.data.image import Image
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import multiprocessing
import matplotlib.pyplot as plt

T1_VALS = {
    'wm': 1.0,
    'gm': 1.3,
    'csf': 4.3
}
BAND_RANGE = {
    'wm': range(1, 5),
    'gm': range(1, 5),
    'csf': [3, ],
    'combined': range(1, 5)
}
PLOT_LIMS = {
    'wm': 1000,
    'gm': 1000,
    'csf': 1500,
    'combined': 1000
}
# scan parameters
slice_in_band = np.tile(np.arange(0, 10), 6).reshape(1, 1, -1)
slicedt = 0.059
def slicetime_correction(image, tissue, tr):
    """
    Rescale data to the given TR to account for T1 relaxation.
    """
    slice_times = tr + (slice_in_band * slicedt)
    denominator = 1 - np.exp(-slice_times/T1_VALS[tissue])
    numerator = 1 - np.exp(-tr/T1_VALS[tissue])
    rescaled_image = image * (numerator / denominator)
    return rescaled_image

def undo_st_correction(rescaled_image, tissue, ti):
    slice_times = 8 + (slice_in_band * slicedt)
    numerator = 1 - np.exp(-slice_times/T1_VALS[tissue])
    denominator = 1 - np.exp(-ti/T1_VALS[tissue])
    descaled_image = rescaled_image * (numerator / denominator)
    return descaled_image

def fit_linear_model(slice_means, method='separate', resolution=10000):
    X = np.arange(0, 10, 1).reshape(-1, 1)
    scaling_factors = np.ones((86, 86, 60))
    y_pred = np.zeros(shape=(resolution*6, 1))
    if method == 'separate':
        X_pred = np.arange(0, 10, 10/resolution).reshape(-1, 1)
        for band in range(1, 5):
            y = slice_means[10*band : 10*band+10]
            model = LinearRegression()
            model.fit(X, y)
            scaling_factors[:, :, 10*band : 10*(band+1)] = model.intercept_ / model.predict(X)
            y_pred[resolution*band : resolution*(band+1), 0] = model.predict(X_pred)
    elif method=='together':
        X_pred = np.tile(np.arange(0, 10, 10/resolution), 4)[..., np.newaxis]
        y_train = np.vstack(np.split(slice_means, 6)[1:5])
        y_train = y_train.mean(axis=0).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y_train)
        scaling_factors[:, :, 10:50] = model.intercept_ / model.predict(np.tile(X, (4, 1))).flatten()
        y_pred[resolution : resolution*5] = model.predict(X_pred)
    return scaling_factors, X_pred, y_pred

def estimate_mt(subject_dirs, rois=['wm', ], tr=8, method='separate'):
    """
    Estimates the slice-dependent MT effect on the given subject's 
    calibration images. Performs the estimation using a linear 
    model and calculates scaling factors which can be used to 
    correct the effect.
    """
    for tissue in rois:
        # initialise array to store image-level means
        mean_array = np.zeros((60, 2*len(subject_dirs)))
        count_array = np.zeros((60, 2*len(subject_dirs), 2)) # wm and gm
        # iterate over subjects
        for n1, subject_dir in enumerate(subject_dirs):
            print(subject_dir)
            # load subject's json
            json_dict = load_json(subject_dir)
            # calculate mean per slice of masked in both calib images
            masked_names = (
                json_dict[f'calib0_{tissue}_masked'],
                json_dict[f'calib1_{tissue}_masked']
            )
            for n2, masked_name in enumerate(masked_names):
                if tissue == 'combined':
                    gm_masked, wm_masked = masked_name
                    gm_masked_data = slicetime_correction(
                        image=Image(gm_masked).data, 
                        tissue='gm',
                        tr=tr
                    )
                    wm_masked_data = slicetime_correction(
                        image=Image(wm_masked).data, 
                        tissue='wm',
                        tr=tr
                    )
                    masked_data = gm_masked_data + wm_masked_data
                    gm_bin = np.where(gm_masked_data>0, 1, 0)
                    gm_count = np.sum(gm_bin, axis=(0, 1))[..., np.newaxis]
                    wm_bin = np.where(wm_masked_data>0, 1, 0)
                    wm_count = np.sum(wm_bin, axis=(0, 1))[..., np.newaxis]
                    count_array[:, 2*n1 + n2, :] = np.hstack((wm_count, gm_count))
                else:
                    # load masked calibration data
                    masked_data = slicetime_correction(
                        image=Image(masked_name).data,
                        tissue=tissue,
                        tr=tr
                    )
                # find zero indices
                masked_data[masked_data==0] = np.nan
                # calculate slicewise summary stats
                slicewise_mean = np.nanmean(masked_data, axis=(0, 1))
                mean_array[:, 2*n1 + n2] = slicewise_mean

        # calculate non-zero slicewise mean of mean_array
        slice_means = np.nanmean(mean_array, axis=1)

        # calculate slicewise mean of tissue type counts
        count_means = np.nanmean(count_array, axis=1)

        # fit linear models to central 4 bands
        # estimate scaling factors using these models
        scaling_factors, X_pred, y_pred = fit_linear_model(slice_means, method=method)
        # plot slicewise mean signal
        slice_numbers = np.arange(0, 60, 1)
        x_coords = np.arange(0, 60, 10)
        plt.figure(figsize=(8, 4.5))
        plt.scatter(slice_numbers, slice_means)
        plt.scatter(np.arange(0, 60, 0.001), y_pred.flatten(), color='k', s=0.1)
        plt.ylim([0, PLOT_LIMS[tissue]])
        plt.xlim([0, 60])
        plt.title(f'Mean signal per slice in {tissue} across 47 subjects.')
        plt.xlabel('Slice number')
        plt.ylabel('Mean signal')
        for x_coord in x_coords:
            plt.axvline(x_coord, linestyle='-', linewidth=0.1, color='k')
        # save plot
        plt_name = Path().cwd() / f'{tissue}_mean_per_slice.png'
        plt.savefig(plt_name)

        # plot slicewise mean tissue count for WM and GM
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(slice_numbers, count_means[:, 0], c='c', label='WM pve > 70%') # wm mean
        ax.scatter(slice_numbers, count_means[:, 1], c='g', label='GM pve > 70%') # gm mean
        ax.legend()
        for x_coord in x_coords:
            ax.axvline(x_coord, linestyle='-', linewidth=0.1, color='k')
        plt.title('Mean number of voxels per slice with' +
                ' PVE $\geqslant$ 70% across 47 subjects.')
        plt.xlabel('Slice number')
        plt.ylabel('Mean number of voxels with PVE $\geqslant$ 70% in a given tissue')
        plt_name = Path().cwd() / 'mean_voxel_count.png'
        plt.savefig(plt_name)

        # # the scaling factors have been estimated on images which have been 
        # # slice-timing corrected - the scaling factors should hence be 
        # # adjusted to account for this, as they will be applied to images 
        # # which haven't had this correction
        # scaling_factors = undo_st_correction(scaling_factors, tissue, tr)

        for subject_dir in subject_dirs:
            json_dict = load_json(subject_dir)
            # load calibration image
            calib_img = Image(json_dict['calib0_img'])
            # create and save scaling factors image
            scaling_img = Image(scaling_factors, header=calib_img.header)
            scaling_dir = Path(json_dict['calib_dir']) / 'MTEstimation'
            create_dirs([scaling_dir, ])
            scaling_name = scaling_dir / f'MTcorr_SFs_{tissue}.nii.gz'
            scaling_img.save(scaling_name)
            
