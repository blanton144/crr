# encoding: utf-8
#
# setup.py
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from setuptools import setup, find_packages, Extension

import os
import argparse
import sys


# The NAME variable should be of the format "sdss-crr".
# Please check your NAME adheres to that format.
NAME = 'crr'
VERSION = '0.1.0dev'
RELEASE = 'dev' in VERSION

class getPybindInclude(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    https://github.com/pybind/python_example/blob/master/setup.py
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


def getIncludes():
    return [
        'include',
        getPybindInclude(),

    ]


sources = [
    'cextern/cCRR.cpp',
    'cextern/sigma.cpp'
]


extra_compile_args = ["--std=c++11", "-fPIC", "-v", "-O3"]
extra_link_args = None
if sys.platform == 'darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.9']
    extra_link_args = ["-v", '-mmacosx-version-min=10.9']

module = Extension(
    'crr/cCRR',
    include_dirs=getIncludes(),
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    sources=sources
)


def run(packages, install_requires):

    setup(name=NAME,
          version=VERSION,
          license='BSD3',
          description='Covariance Regularized Reconstruction',
          long_description=open('README.rst').read(),
          author='Michael Blanton',
          author_email='michael.blanton@gmail.com',
          keywords='astronomy software',
          ext_modules=[module],
          url='https://github.com/sdss/crr',
          include_package_data=True,
          packages=packages,
          install_requires=install_requires,
          package_dir={'': 'python'},
          scripts=[],
          classifiers=[
              'Development Status :: 4 - Beta',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Natural Language :: English',
              'Operating System :: OS Independent',
              'Programming Language :: Python',
              'Programming Language :: Python :: 2.6',
              'Programming Language :: Python :: 2.7',
              'Topic :: Documentation :: Sphinx',
              'Topic :: Software Development :: Libraries :: Python Modules',
          ],
          )


def get_requirements(opts):
    ''' Get the proper requirements file based on the optional argument '''

    if opts.dev:
        name = 'requirements_dev.txt'
    elif opts.doc:
        name = 'requirements_doc.txt'
    else:
        name = 'requirements.txt'

    requirements_file = os.path.join(os.path.dirname(__file__), name)
    install_requires = [line.strip().replace('==', '>=') for line in open(requirements_file)
                        if not line.strip().startswith('#') and line.strip() != '']
    return install_requires


def remove_args(parser):
    ''' Remove custom arguments from the parser '''

    arguments = []
    for action in list(parser._get_optional_actions()):
        if '--help' not in action.option_strings:
            arguments += action.option_strings

    for arg in arguments:
        if arg in sys.argv:
            sys.argv.remove(arg)


if __name__ == '__main__':

    # Custom parser to decide whether which requirements to install
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument('-d', '--dev', dest='dev', default=False, action='store_true',
                        help='Install all packages for development')
    parser.add_argument('-o', '--doc', dest='doc', default=False, action='store_true',
                        help='Install only core + documentation packages')

    # We use parse_known_args because we want to leave the remaining args for distutils
    args = parser.parse_known_args()[0]

    # Get the proper requirements file
    install_requires = get_requirements(args)

    # Now we remove all our custom arguments to make sure they don't interfere with distutils
    remove_args(parser)

    # Have distutils find the packages
    packages = find_packages(where='python')

    # Runs distutils
    run(packages, install_requires)
