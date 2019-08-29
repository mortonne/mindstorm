import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mindstorm",
    version="0.0.1",
    author="Neal Morton",
    author_email="mortonne@gmail.com",
    description="Advanced analysis of neuroimaging data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mortonne/mindstorm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ]
)
