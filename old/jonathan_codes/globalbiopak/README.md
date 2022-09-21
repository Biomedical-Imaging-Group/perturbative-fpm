# GlobalBioPak

## General information

*Temporary, 2021/07/20*

Welcome to this (internal for now) Git for GlobalBioPak, a library inspired by GlobalBioIm! 
We will build it incrementally, as we write codes for our projects in this unified framework. 

It is based on the following principles:
- Pytorch: modular codes to run on CPU and GPU
- Application-oriented: we do not try to replicate the whole GlobalBioIm library, but implement codes for our projects

For now, the library has a quick implementation of:
- Basic LinOp class
- LinOp3D class based on Tomosipo, a Pytorch wrapper of the ASTRA library (https://github.com/ahendriksen/tomosipo)
- PhaseRetrieval with a few algorithms to solve y=|Ax|^2 efficiently
- Cam to add noise to images
- utilsEM with a few functions useful in cryoEM

Everyone interested to contribute is welcome, just send a message to Pakshal Bohra or Jonathan Dong.

## Installation

To install this repository, use the terminal, go to this package folder, and run the following command: 

``` pip install -e . ```

More info on https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/

## Git repository

To unify the git commit syntax, let's try to follow this example: "Write commits in present tense, starting with a capitalized letter" (More info on https://chris.beams.io/posts/git-commit/)

We can all make changes to the master branch. Maybe it will become a mess but we'll fix that later ;)

If you're not familiar with Git, we can see that together.

## Structure

*Based on a simplified version of https://drivendata.github.io/cookiecutter-data-science/*

```
├── README.md          <- The top-level README for developers using this project.
│
├── globalbiopak       <- Source code for use in this project.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── data
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
```