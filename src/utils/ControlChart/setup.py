from setuptools import setup, Extension
import pybind11
import sysconfig

ext_modules = [
    Extension(
        "permutation_test",
        ["permutation_test.cpp"],
        include_dirs=[pybind11.get_include(), sysconfig.get_path("include")],
        language="c++",
        extra_compile_args=["/O2"], 
    ),
]

setup(
    name="permutation_test",
    version="0.1",
    ext_modules=ext_modules,
)
