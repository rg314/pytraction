[![Build Status](https://travis-ci.com/rg314/pytraction.svg?token=BCkcrsWckKEnE7AqL2uD&branch=main)](https://travis-ci.com/rg314/pytraction)
[![codecov](https://codecov.io/gh/rg314/pytraction/branch/main/graph/badge.svg?token=5HLPLUWIXN)](https://codecov.io/gh/rg314/pytraction)

<p align="center">
<img src="https://user-images.githubusercontent.com/35999546/112598957-2fa21a00-8e07-11eb-847c-37f311e4c919.png" alt="PyTraction">
</p>



Bayesian Traction Force Microscopy

**Motivation**: TFM data is annoying, disperse tool set (all in matlab / ImageJ / javascript)

**PyTraction**: Modern software and easy to use end-to-end


## TO-DO

:clock9: Tests for edge cases on input

:clock9: Save to hdf5 rather than csv

:clock9: Simple hdf5 usage

:clock9: Write manuscript


## Colab examples
Please try running the following notebooks on google colab. You will need to generate a [personal access token](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token).

[Example 1: Basic usage with an image stack and reference image in the correct format](https://colab.research.google.com/github/rg314/pytraction/blob/main/examples/example1.ipynb)

[Example 2: Basic usage with an ROI](https://colab.research.google.com/github/rg314/pytraction/blob/main/examples/example2.ipynb)

[Example 3: Basic usage unexpected image formats](https://colab.research.google.com/github/rg314/pytraction/blob/main/examples/example3.ipynb)


### TL;DR
Navigate to folder where you want to install and ensure you have [miniconda](https://docs.conda.io/en/latest/miniconda.html) and [git]( https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed and run the following commands.

```
git clone https://github.com/rg314/pytraction.git
conda create -n pytraction python=3.8
conda activate pytraction
pip install pytraction/
cd pytraction
python scripts/get_example_data.py
pip install notebook
python -m ipykernel install --user --name=pytraction
jupyter notebook scripts/usage.ipynb
```

## Draft manuscript

Please follow [OneDrive link to draft manuscript](https://universityofcambridgecloud-my.sharepoint.com/:w:/g/personal/rdg31_cam_ac_uk/Ed0Z-nD1hrhMuCujn5yhrdoBu4-VcEIdpUdaSLyZo4KLTA?e=IYqPJC) or [OneDrive link to folder](https://universityofcambridgecloud-my.sharepoint.com/:f:/g/personal/rdg31_cam_ac_uk/EldvnfWg5k1NsGt5L7bthSYBFdrhbKrX1aaTAoxSKeag9g). I've chosen OneDrive as it nicely integrates with EndNote and multiple users editing at once.

The draft manuscript has only been sent to a few authors to date. Authors are listed alphabetically. Contributors will be invited to manuscript if a significant contribution is made which can consist of:
-	Significantly contributing to design or code of the core python package.
-	Demonstrating usage of core python package compared to other analysis techniques.
-	Demonstrating usage of core python package with computational techniques.
-	Fixing major bugs in core python package.
-	Contributed or provided experimental data for the manuscript.

If you believe that you have made a significant contribution (and have not been invited to the manuscript) or would like to make a significant contribution please [contact me](https://github.com/rg314).


## Installation

**Note**: if running on Windows you may have problems with the shapely library and some ctypes extensions. Please install shapely via `conda install shapely`. 

For HTTPS
```git clone https://github.com/rg314/pytraction.git```

For SSH
```git clone git@github.com:rg314/pytraction.git```


Create conda env

```
conda create -n pytraction python=3.8
```

Activate conda env

```
conda activate pytraction
```

Install pytraction (for dev install in editable mode `- e .`)

```
pip install pytraction/
```

Change directory to repository

```
cd pytraction
```

To get example data run

```
python scripts/get_example_data.py
```

Install jupyter notebook

```
pip install notebook
```

Install pytraction kernel to use with jupyter notebook

```
python -m ipykernel install --user --name=pytraction
```

For basic usage run

```
jupyter notebook scripts/usage.ipynb
```


## Example

The following code show an basic example. Please make sure you download example data by running the following script `python scripts/get_example_data.py`. You need to make sure that the `data` folder is in your working directory when you run the following code. For a more in-depth examples please see [scripts/usage.ipynb](https://github.com/rg314/pytraction/blob/main/scripts/usage.ipynb)


For basic usage:

```
from pytraction import TractionForceConfig
from pytraction import plot, prcoess_stack

pix_per_mu = 1.3 # The number of pixels per micron 
E = 100 # Youngs modulus in Pa

img_path = 'data/example1/e01_pos1_axon1.tif'
ref_path = 'data/example1/e01_pos1_axon1_ref.tif'

traction_config = TractionForceConfig(pix_per_mu, E=E)
img, ref, _ = traction_config.load_data(img_path, ref_path)
log = process_stack(img, ref)

plot(log, frame=0)
plt.show()
```

![image](https://user-images.githubusercontent.com/35999546/111919773-962fdc80-8a83-11eb-9230-ec9e588a9b77.png)


### Contributing
Contributing: Think of [Big O notation](https://en.wikipedia.org/wiki/Big_O_notation#:~:text=Big%20O%20notation%20is%20a,a%20particular%20value%20or%20infinity.) and [Occam's_razor](https://en.wikipedia.org/wiki/Occam%27s_razor)

For commiting big code blocks please relate them to issues and create a new branch. The branch name as the abbriviated issue (issue8 = iss8). 

```git checkout -b iss8```

Commit to current branch and assign a reviewer when merging pull request into main branch from the webapp.


## Amazing references
Here is a list of common resources that have been used

1. [Imagej piv](https://sites.google.com/site/qingzongtseng/piv)
2. [Easy-to-use_TFM_package](https://github.com/CellMicroMechanics/Easy-to-use_TFM_package)
3. [openpiv-python](http://www.openpiv.net/openpiv-python/)
4. [opencv-python](https://opencv-python-tutroals.readthedocs.io/en/latest/index.html)
5. [A great paper from Sabass lab](https://www.nature.com/articles/s41598-018-36896-x)
6. [Template matching](https://sites.google.com/site/qingzongtseng/template-matching-ij-plugin/tuto2)

