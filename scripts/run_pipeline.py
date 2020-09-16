"""
This script performs the full minimal pre-processing ASL pipeline 
for the Human Connectome Project (HCP) ASL data.

This currently requires that the script is called followed by 
the directories of the subjects of interest and finally the 
name of the MT correction scaling factors image.
"""

import sys
import os

from hcpasl.initial_bookkeeping import initial_processing
from hcpasl.m0_mt_correction import correct_M0
from hcpasl.asl_correction import hcp_asl_moco
from hcpasl.asl_differencing import tag_control_differencing
from hcpasl.asl_perfusion import run_fabber_asl, run_oxford_asl
from hcpasl.projection import project_to_surface
from pathlib import Path
import subprocess
import argparse
from multiprocessing import cpu_count

def process_subject(subject_dir, mt_factors, cores, order, mbpcasl, structural, surfaces, fmaps, gradients=None):
    """
    Run the hcp-asl pipeline for a given subject.

    Parameters
    ----------
    subject_dir : str
        Path to the subject's base directory.
    mt_factors : str
        Path to a .txt file of pre-calculated MT correction 
        factors.
    cores : int
        Number of cores to use.
        When applying motion correction, this is the number 
        of cores that will be used by regtricks.
    order : int
        The interpolation order to use for registrations.
        Regtricks passes this on to scipy's map_coordinates. 
        The meaning of the value can be found in the scipy 
        documentation.
    mbpcasl : str
        Path to the subject's mbPCASL sequence.
    structural : dict
        Contains the locations of important structural files.
    surfaces : dict
        Contains the locations of the surfaces needed for the 
        pipeline.
    fmaps : dict
        Contains the locations of the fieldmaps needed for 
        distortion correction.
    gradients : str, optional
        Path to a gradient coefficients file for use in 
        gradient distortion correction.
    """
    subject_dir = Path(subject_dir)
    mt_factors = Path(mt_factors)
    initial_processing(subject_dir, mbpcasl=mbpcasl, structural=structural, surfaces=surfaces)
    correct_M0(subject_dir, mt_factors)
    hcp_asl_moco(subject_dir, mt_factors, cores=cores, order=order)
    for target in ('asl', 'structural'):
        dist_corr_call = [
            "hcp_asl_distcorr",
            str(subject_dir.parent),
            subject_dir.stem,
            "--target",
            target,
            "--fmap_ap",
            fmaps['AP'],
            "--fmap_pa",
            fmaps['PA']
        ]
        if gradients:
            dist_corr_call.append('--grads')
            dist_corr_call.append(gradients)
        subprocess.run(dist_corr_call, check=True)
        if target == 'structural':
            pv_est_call = [
                "pv_est",
                str(subject_dir.parent),
                subject_dir.stem
            ]
            subprocess.run(pv_est_call, check=True)
        tag_control_differencing(subject_dir, target=target)
        if target == 'asl':
            run_oxford_asl(subject_dir, target=target)
            project_to_surface(subject_dir, target=target)
        else:
            surface_proc_call = [
                "hcp_asl_projected",
                str(subject_dir)
            ]
            subprocess.run(surface_proc_call, check=True)

def main():
    """
    Main entry point for the hcp-asl pipeline.
    """
    # argument handling
    parser = argparse.ArgumentParser(
        description="This script performs the minimal processing for the "
                    + "HCP-Aging ASL data.")
    parser.add_argument(
        "subject_dir",
        help="The directory of the subject you wish to process."
    )
    parser.add_argument(
        "scaling_factors",
        help="Filename of the empirically estimated MT-correction"
            + "scaling factors."
    )
    parser.add_argument(
        "-g",
        "--grads",
        help="Filename of the gradient coefficients for gradient"
            + "distortion correction (optional)."
    )
    parser.add_argument(
        "-s",
        "--struct",
        help="Filename for the acpc-aligned, dc-restored structural image."
    )
    parser.add_argument(
        "--sbrain",
        help="Filename for the brain-extracted acpc-aligned, "
            + "dc-restored structural image."
    )
    parser.add_argument(
        "--lmid",
        help="Filename for the left mid surface."
    )
    parser.add_argument(
        "--rmid",
        help="Filename for the right mid surface."
    )
    parser.add_argument(
        "--lwhite",
        help="Filename for the left white surface."
    )
    parser.add_argument(
        "--rwhite",
        help="Filename for the right white surface."
    )
    parser.add_argument(
        "--lpial",
        help="Filename for the left pial surface."
    )
    parser.add_argument(
        "--rpial",
        help="Filename for the right pial surface."
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Filename for the mbPCASLhr acquisition."
    )
    parser.add_argument(
        "--fmap_ap",
        help="Filename for the AP fieldmap for use in distortion correction"
    )
    parser.add_argument(
        "--fmap_pa",
        help="Filename for the PA fieldmap for use in distortion correction"
    )
    parser.add_argument(
        "-c",
        "--cores",
        help="Number of cores to use for registration operations. "
            + f"Your PC has {cpu_count()}. Default is 1.",
        default=1,
        type=int
    )
    parser.add_argument(
        "--interpolation",
        help="Interpolation order for registrations. Default is 3.",
        default=3,
        type=int
    )
    parser.add_argument(
        "--fabberdir",
        help="User Fabber executable in <fabberdir>/bin/ for users"
            + "with FSL < 6.0.4"
    )
    # assign arguments to variables
    args = parser.parse_args()
    mt_name = args.scaling_factors
    subject_dir = args.subject_dir
    structural = {'struct': args.struct, 'sbrain': args.sbrain}
    surfaces = {
        'L_mid': args.lmid, 'R_mid': args.rmid,
        'L_white': args.lwhite, 'R_white':args.rwhite,
        'L_pial': args.lpial, 'R_pial': args.rpial
    }
    mbpcasl = args.input
    fmaps = {'AP': args.fmap_ap, 'PA': args.fmap_pa}
    cores = args.cores
    order = args.interpolation
    if args.fabberdir:
        if not os.path.isfile(os.path.join(args.fabberdir, "bin", "fabber_asl")):
            print("ERROR: specified Fabber in %s, but no fabber_asl executable found in %s/bin" % (args.fabberdir, args.fabberdir))
            sys.exit(1)

        # To use a custom Fabber executable we set the FSLDEVDIR environment variable
        # which prioritises executables in $FSLDEVDIR/bin over those in $FSLDIR/bin.
        # Note that this could cause problems in the unlikely event that the user
        # already has a $FSLDEVDIR set up with custom copies of other things that
        # oxford_asl uses...
        print("Using Fabber-ASL executable %s/bin/fabber_asl" % args.fabberdir)
        os.environ["FSLDEVDIR"] = os.path.abspath(args.fabberdir)

    print(f"Processing subject {subject_dir}.")
    if args.grads:
        print("Including gradient distortion correction step.")
        process_subject(subject_dir=subject_dir,
                        mt_factors=mt_name,
                        cores=cores,
                        order=order,
                        gradients=args.grads,
                        mbpcasl=mbpcasl,
                        structural=structural,
                        surfaces=surfaces,
                        fmaps=fmaps
                        )
    else:
        print("Not including gradient distortion correction step.")
        process_subject(subject_dir=subject_dir,
                        mt_factors=mt_name,
                        cores=cores,
                        order=order,
                        mbpcasl=mbpcasl,
                        structural=structural,
                        surfaces=surfaces,
                        fmaps=fmaps
                        )

if __name__ == '__main__':
    main()