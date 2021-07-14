
from setuptools import setup


VERSION = '0.0.0'


DESCRIPTION = 'Bayesian Traction Force Microscopy'


CLASSIFIERS = [
    'Natural Language :: English',
    'Programming Language :: Python :: 3.8',
]

REQUIREMENTS = [
    'attrs==20.3.0',
    'certifi==2020.12.5',
    'cycler==0.10.0',
    'Cython==0.29.22',
    'decorator==4.4.2',
    'gitdb==4.0.5',
    'GitPython==3.1.14',
    'imageio==2.9.0',
    'iniconfig==1.1.1',
    'kiwisolver==1.3.1',
    'matplotlib==3.3.4',
    'natsort==7.1.1',
    'networkx==2.5',
    'numpy==1.20.1',
    'opencv-python==4.5.1.48',
    'OpenPIV==0.23.4',
    'packaging==20.9',
    'pandas==1.2.3',
    'Pillow>=8.2.0',
    'pluggy==0.13.1',
    'py==1.10.0',
    'pyparsing==2.4.7',
    'pytest==6.2.2', 
    'python-dateutil==2.8.1',
    'pytz==2021.1',
    'PyWavelets==1.1.1',
    'read-roi==1.6.0',
    'scikit-image==0.18.1',
    'scipy==1.6.1',
    'Shapely==1.7.1',
    'smmap==3.0.5',
    'toml==0.10.2',
    'tqdm==4.59.0',
    'albumentations==0.5.2',
    'segmentation_models_pytorch==0.2.0',
    'shapely==1.7.1',
    'googledrivedownloader==0.4',
    'requests==2.25.1',
    'scikit-learn==0.24.1',
    'h5py==3.2.1',
]


SETUP_REQUIRES = ('pytest-cov', 'pytest-runner','pytest', 'codecov')
TESTS_REQUIRES = ('pytest-cov','codecov')

 

PACKAGES = [
    'pytraction',
    'pytraction.net',
]


options = {
    'name': 'pytraction',
    'version': VERSION,
    'author': 'Ryan Greenhalgh',
    'author_email': 'rdg31@cam.ac.uk',
    'description': DESCRIPTION,
    'classifiers': CLASSIFIERS,
    'packages': PACKAGES,
    'setup_requires': SETUP_REQUIRES,
    'test_requires': TESTS_REQUIRES,
    'install_requires': REQUIREMENTS,
    'entry_points': {
            'console_scripts': ["pytraction_get_data=pytraction.get_example_data:main"]},
}
setup(**options)



