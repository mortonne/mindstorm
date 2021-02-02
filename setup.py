import setuptools
import glob

scripts = glob.glob('scripts/*')
setuptools.setup(scripts=scripts)
