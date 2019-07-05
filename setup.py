import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pt-trainer",
    version="0.0.2",
    author="Ilja Manakov",
    author_email="ilja.manakov@gmx.de",
    description="A toolkit for training pytorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IljaManakov/PyTorch-Trainer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
