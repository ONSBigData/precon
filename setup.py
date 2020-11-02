#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://precon.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='precon',
    version='0.6.2',
    description='A set of functions to calculate Prices Economics statistics.',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Mitchell Edmunds',
    author_email='mitchell.edmunds@ext.ons.gov.uk',
    url='https://github.com/ONSBigData/precon',
    packages=[
        'precon',
    ],
    package_dir={'precon': 'precon'},
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
    ],
    license='MIT',
    zip_safe=False,
    keywords='precon',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
