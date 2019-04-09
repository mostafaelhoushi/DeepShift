#!/usr/bin/python3
# Installation/setup script for Simple Python Fixed-Point Module
# RW Penney, January 2007

from distutils.core import setup
from FixedPoint import SPFPM_VERSION

setup(
    author = 'RW Penney',
    author_email = 'rwpenney@users.sourceforge.net',
    description = 'Tools for arithmetic on fixed-point (binary) numbers',
    fullname = 'Simple Python Fixed-Point Module',
    keywords = 'arithmetic, fixed-point, trigonometry, arbitrary precision',
    license = 'PSF Python License',
    long_description = \
        'The Simple Python Fixed Point Module (SPFPM) ' + \
        'is a pure-Python module which provides basic facilities ' + \
        'for manipulating binary fixed-point numbers ' + \
        'of essentially arbitrary precision. ' + \
        'It aims to be more suitable for simulating digital ' + \
        'fixed-point artihmetic within electronic hardware ' + \
        '(e.g. for digital signal processing (DSP) applications) ' + \
        'than the Decimal package, which is more concerned ' + \
        'with fixed-point arithmetic in base-10. ' + \
        'SPFPM supports basic arithmetic operations as well as a range ' + \
        'of mathematical functions including sqrt, exp, sin, cos, atan etc.',
    name = 'spfpm',
    url = 'https://github.com/rwpenney/spfpm',
    version = SPFPM_VERSION,
    py_modules = [ 'FixedPoint' ],
    classifiers = [ 'Programming Language :: Python :: 3',
                    'Intended Audience :: Science/Research',
                    'Operating System :: OS Independent',
                    'Topic :: Scientific/Engineering :: Mathematics',
                    'Topic :: Software Development :: Embedded Systems' ]
)
