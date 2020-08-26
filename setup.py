import setuptools
import PyMMS

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyTST",
    version=PyMMS.__version__,
    author="Sebastien Lemaire",
    author_email="sebastien.lemaire@soton.ac.uk",
    description="Tools performing Transient Scanning Technique",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WavEC-Offshore-Renewables/pyTST",
    packages=setuptools.find_packages(),
    scripts=['TST-cli'],
    requires=['Numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires='>=3.2',
)
