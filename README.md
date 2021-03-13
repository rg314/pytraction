[![Build Status](https://travis-ci.com/rg314/pytraction.svg?token=BCkcrsWckKEnE7AqL2uD&branch=main)](https://travis-ci.com/rg314/pytraction)

# pytraction
Bayesian Traction Force Microscopy


#### TO-DO

:heavy_check_mark: Base PIV

:heavy_check_mark: Allign stacks

:clock9: Finish writing bayesian TFM functions [from matlab implementation](https://github.com/CellMicroMechanics/Easy-to-use_TFM_package)

- [ ] train vanilla efficentnet CNN to get cell outline

- [ ] implement NMT for filtering U, V vectors

- [ ] end-to-end testing


## Installation
I've not yet included the reqs so this will be broken and you'll need to install a few libs manually for now

```git clone git@github.com:rg314/pytraction.git```


Create conda env

```
conda create -n pytraction python=3.8
```

Install autoballs in editable mode

```
pip install -e .
```

## Amazing references
Here is a list of common resources that have been used

1. [Imagej piv](https://sites.google.com/site/qingzongtseng/piv)
2. [Easy-to-use_TFM_package](https://github.com/CellMicroMechanics/Easy-to-use_TFM_package)
3. [openpiv-python](http://www.openpiv.net/openpiv-python/)
4. [opencv-python](https://opencv-python-tutroals.readthedocs.io/en/latest/index.html)
5. [A great paper from Sabass lab](https://www.nature.com/articles/s41598-018-36896-x)
6. [Template matching](https://sites.google.com/site/qingzongtseng/template-matching-ij-plugin/tuto2)

