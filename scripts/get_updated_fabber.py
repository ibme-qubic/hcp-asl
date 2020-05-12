#!/usr/bin/env python
"""
This is a simple script to download an appropriate updated copy of
Fabber to use in the pipeline, prior to the release of FSL 6.0.4
"""
import os
import sys
import stat

import requests

def main():
    if len(sys.argv) != 2:
        print("Usage: get_updated_fabber <destination dir>")
        sys.exit(1)

    destdir = sys.argv[1]
    if not destdir.rstrip("/").endswith("bin"):
        destdir = os.path.join(destdir, "bin")

    if sys.platform == "linux":
        print("Using Linux executable built under Ubuntu 18.04.")
        print("This may work for other binary compatible Linux distros")
        print("including recent Centos")
        url = "https://github.com/ibme-qubic/fabber_models_asl/releases/download/v2.0.3/fabber_asl_ubuntu18"
    elif sys.platform == "darwin":
        print("Using Linux executable built for Mac OSX")
        url = "https://github.com/ibme-qubic/fabber_models_asl/releases/download/v2.0.3/fabber_asl_mac"
    else:
        print("Unsupported platform: %s - cannot download Fabber" % sys.platform)
        sys.exit(1)

    print("Downloading %s" % url)
    os.makedirs(destdir, exist_ok=True)
    src = requests.get(url, allow_redirects=True)
    dest = os.path.join(destdir, "fabber_asl")
    with open(dest, "wb") as dest_file:
        dest_file.write(src.content)
    os.chmod(dest, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print("Downloaded fabber to %s" % dest)

    # Check the executable actually works
    retcode=os.system("%s --version" % dest)
    if retcode != 0:
        print("ERROR: downloaded executable did not run correctly - check if your platform is compatible")
        sys.exit(1)
    print("Executable ran successfully")
    fabberdir = os.path.abspath(destdir.rstrip("/").rstrip("bin").rstrip("/"))
    print("To use in the HCP-ASL pipeline add the option --fabberdir=%s" % fabberdir)

if __name__ == '__main__':
    main()