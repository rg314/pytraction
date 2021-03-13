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

