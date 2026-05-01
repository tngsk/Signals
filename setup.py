import os
import sys
import pybind11
from setuptools import setup, Extension

# The sdist doesn't inherently include RNBO_Integration if it's not a python module
# By resolving the path absolutely to where setup.py runs, we can include it correctly.

curr_dir = os.path.dirname(os.path.abspath(__file__))
rnbo_inc = os.path.join(curr_dir, "RNBO_Integration")
rnbo_common_inc = os.path.join(rnbo_inc, "common")

ext_modules = [
    Extension(
        "signals.rnbo_osc",
        ["src/signals/rnbo_osc.cpp"],
        include_dirs=[
            pybind11.get_include(),
            rnbo_inc,
            rnbo_common_inc
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14"],
    ),
]

setup(
    ext_modules=ext_modules,
)
