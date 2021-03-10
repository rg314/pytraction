from setuptools import setup


VERSION = '0.0.0'


DESCRIPTION = 'Bayesian Traction Force Microscopy'


CLASSIFIERS = [
    'Natural Language :: English',
    'Programming Language :: Python :: 3.8',
]

REQUIREMENTS = []


SETUP_REQUIRES = ('') #('pytest-cov', 'pytest-runner','pytest', 'codecov')
TESTS_REQUIRES = ('') #('pytest-cov','codecov')


PACKAGES = [
    'pytraction',
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
    'package_data': DATA,
}
setup(**options)
