#!/usr/bin/env python
# -*- coding:utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Distutils import build_ext


ext_modules = [
    Extension("preimage.kernels._generic_string", ["preimage/kernels/_generic_string.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("preimage.inference._branch_and_bound", ["preimage/inference/_branch_and_bound.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("preimage.inference._bound_calculator", ["preimage/inference/_bound_calculator.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("preimage.inference._node", ["preimage/inference/_node.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("preimage.inference._node_creator", ["preimage/inference/_node_creator.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("preimage.features.gs_similarity_weights", ["preimage/features/gs_similarity_weights.pyx"],
              include_dirs=[numpy.get_include()])
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)