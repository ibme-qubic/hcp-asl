from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='hcpasl',
    version='0.0.1',
    author='Jack Toner',
    author_email='jack.toner@nottingham.ac.uk',
    description='Minimal ASL processing pipeline for the HCP.',
    long_description=long_description,
    url='https://github.com/ibme-qubic/hcp-asl',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'fslpy',
        'pyfab',
        'nibabel',
        'regtools @ git+https://github.com/tomfrankkirk/regtools.git',
        'toblerone @ git+https://github.com/tomfrankkirk/toblerone.git',
        'gradunwarp @ git+https://github.com/Washington-University/gradunwarp.git'
    ],
    entry_points={
        'console_scripts': [
            'hcp_asl = scripts.run_pipeline:main',
            'hcp_asl_distcorr = scripts.distcorr_warps:main',
            'get_updated_fabber = scripts.get_updated_fabber:main',
        ]
    }
)
