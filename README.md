[![Build Status](https://travis-ci.com/rg314/pytraction.svg?token=BCkcrsWckKEnE7AqL2uD&branch=main)](https://travis-ci.com/rg314/pytraction)
[![codecov](https://codecov.io/gh/rg314/pytraction/branch/main/graph/badge.svg?token=5HLPLUWIXN)](https://codecov.io/gh/rg314/pytraction)


# pytraction
Bayesian Traction Force Microscopy

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1TG-_TGr27_W_rScw3IQgroe8SP0Wqdts" alt="Example logo" width="800"/>
</p>

**Motivation**: TFM data is annoying, disperse tool set (all in matlab / java / javascript)

**PyTraction**: Modern software and easy to use end-to-end

**Contributing**: Think of [Big O notation](https://en.wikipedia.org/wiki/Big_O_notation#:~:text=Big%20O%20notation%20is%20a,a%20particular%20value%20or%20infinity.) and [Occam's_razor](https://en.wikipedia.org/wiki/Occam%27s_razor)


#### TO-DO

:heavy_check_mark: Base PIV

:heavy_check_mark: Allign stacks

:heavy_check_mark: Finish writing bayesian TFM functions [from matlab implementation](https://github.com/CellMicroMechanics/Easy-to-use_TFM_package)

:heavy_check_mark: implement NMT for filtering U, V vectors

:heavy_check_mark: train vanilla efficentnet CNN to get cell outline

:heavy_check_mark: end-to-end testing

:clock9: create a list of common bugs and raise sort errors out

:clock9: refactor code and slim down into general use API

## Installation
For HTTPS
```git clone https://github.com/rg314/pytraction.git```

For SSH
```git clone git@github.com:rg314/pytraction.git```


Create conda env

```
conda create -n pytraction python=3.8
```

Install pytraction (for dev install in editable mode `- e .`)

```
pip install pytraction/
```

To get example data run

```
python scripts/get_example_data.py
```

For basic usage run

```
python scripts/usage.py
```


### Contributing
Contributing: Think of [Big O notation](https://en.wikipedia.org/wiki/Big_O_notation#:~:text=Big%20O%20notation%20is%20a,a%20particular%20value%20or%20infinity.) and [Occam's_razor](https://en.wikipedia.org/wiki/Occam%27s_razor)

For commiting big code blocks please relate them to issues and create a new branch. The branch name as the abbriviated issue (issue8 = iss8). 

```git checkout -b iss8```

Commit to current branch and assign a reviewer when merging pull request into main branch from the webapp.

## Example
From inital testing it is possible to pass input x,y,u,v into the function and the matlab implementation can be reproduced

Output:

![image](https://user-images.githubusercontent.com/35999546/111041793-ee3a6380-8431-11eb-906f-6698aaa6ba03.png)

L = 141.3021 for matlab
L = 138.9749 for python

## Amazing references
Here is a list of common resources that have been used

1. [Imagej piv](https://sites.google.com/site/qingzongtseng/piv)
2. [Easy-to-use_TFM_package](https://github.com/CellMicroMechanics/Easy-to-use_TFM_package)
3. [openpiv-python](http://www.openpiv.net/openpiv-python/)
4. [opencv-python](https://opencv-python-tutroals.readthedocs.io/en/latest/index.html)
5. [A great paper from Sabass lab](https://www.nature.com/articles/s41598-018-36896-x)
6. [Template matching](https://sites.google.com/site/qingzongtseng/template-matching-ij-plugin/tuto2)

