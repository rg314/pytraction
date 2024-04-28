import numpy as np
import os

from shapely import Polygon
from pytraction.core import TractionForceConfig
from pytraction.noise import get_noise


def test_noise():
    config_path = os.path.join("example_data", "example_config.yaml")

    # Create config instance
    config = TractionForceConfig(
        config_path=config_path,
        elastic_modulus=100,
        scaling_factor=1.3,
        poisson_ratio=0.48,
        window_size=16,
        meshsize=8
    )

    # Create a 2D array filled with random values drawn from a Gaussian distribution
    std_dev_a = 10
    std_dev_b = 20

    noise_a = np.random.normal(loc=0, scale=std_dev_a, size=(100, 100))
    noise_b = np.random.normal(loc=0, scale=std_dev_b, size=(100, 100))

    noise = np.block([[noise_a, noise_a, noise_a, noise_a],
                      [noise_a, noise_b, noise_b, noise_a],
                      [noise_a, noise_b, noise_b, noise_a],
                      [noise_a, noise_a, noise_a, noise_a]])

    x_val, y_val = np.linspace(-20, 20, 400), np.linspace(-20, 20, 400)
    xx, yy = np.meshgrid(x_val, y_val)

    # Test: Border noise
    noise_calc = get_noise(x=xx, y=yy, u=noise, v=noise, polygon=None, noise=100)
    assert np.isclose(noise_calc, 1 / (std_dev_a ** 2), rtol=0.1)

    # Test: Individual noise
    noise_calc = get_noise(x=xx, y=yy, u=noise, v=noise, polygon=None,
                           noise=np.array([noise_a.flatten(), noise_a.flatten()]).flatten())
    assert np.isclose(noise_calc, 1 / (std_dev_a ** 2), rtol=0.1)

    # Test: Polygon noise
    coords = ((-10., -10.), (-10., 10.), (10., 10.), (-10., 10.), (-10., -10.))
    poly = Polygon(coords)

    noise_calc = get_noise(x=xx, y=yy, u=noise, v=noise, polygon=None, noise=poly)
    assert np.isclose(noise_calc, 1 / (std_dev_b ** 2), rtol=0.1)

    noise_calc = get_noise(x=xx, y=yy, u=noise, v=noise, polygon=poly, noise=0)
    assert np.isclose(noise_calc, 1 / (std_dev_a ** 2), rtol=0.1)
