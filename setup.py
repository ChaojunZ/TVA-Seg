""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path,getcwd
here = path.abspath(path.dirname(__file__))

exec(open('tva_seg/src/open_clip/version.py').read())
setup(
    name='open_clip_torch',
    version=__version__,
    description='OpenCLIP',
    url='***/open_clip', # Please replace *** with the OpenCLIP link
    author='',
    author_email='',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='CLIP pretrained',
    package_dir={'': 'tva_seg/src'},
    packages=find_packages(where='tva_seg/src'),
    include_package_data=True,
    python_requires='>=3.7',
)