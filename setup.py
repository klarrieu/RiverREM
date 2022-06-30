import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RiverREM",
    version="0.0.1",
    author="Kenneth Larrieu",
    author_email="kennylarrieu@gmail.com",
    description="Make river relative elevation models (REM) and REM visualizations from an input digital elevation model (DEM).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/klarrieu/RiverREM",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)