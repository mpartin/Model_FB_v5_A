from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl
import numpy as np

setup(
      include_dirs = [np.get_include()],
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("Model_FB_cython",
                               ["Model_FB.pyx"],
                               libraries=cython_gsl.get_libraries(),
                               library_dirs=[cython_gsl.get_library_dir()],
                               include_dirs=[cython_gsl.get_cython_include_dir()])]
      )