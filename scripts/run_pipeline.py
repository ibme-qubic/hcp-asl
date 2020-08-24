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

def process_subject(subject_dir, mt_factors, cores, order, gradients=None):
    """
    Run pipeline for individual subject specified by 
    `subject_dir`.
    """
    subject_dir = Path(subject_dir)
    mt_factors = Path(mt_factors)
    initial_processing(subject_dir)
    correct_M0(subject_dir, mt_factors)
    hcp_asl_moco(subject_dir, mt_factors, cores=cores, order=order)
    for target in ('asl', 'structural'):
        dist_corr_call = [
            "hcp_asl_distcorr",
            str(subject_dir.parent),
            subject_dir.stem,
            "--target",
            target
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
        "-c",
        "--cores",
        help="Number of cores to use for registration operations. "
            + f"Your PC has {cpu_count()}. Default is 1.",
        default=1,
        type=int
    )
    parser.add_argument(
        "-i",
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
        process_subject(subject_dir, mt_name, cores, order, args.grads)
    else:
        print("Not including gradient distortion correction step.")
        process_subject(subject_dir, mt_name, cores, order)

if __name__ == '__main__':
    main()