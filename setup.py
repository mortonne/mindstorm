import setuptools
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

scripts = glob.glob('scripts/*')

setuptools.setup(
    name="mindstorm",
    version="0.2.1",
    author="Neal Morton",
    author_email="mortonne@gmail.com",
    description="Advanced analysis of neuroimaging data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mortonne/mindstorm",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    scripts=scripts
)
