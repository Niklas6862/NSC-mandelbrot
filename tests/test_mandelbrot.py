from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mandelbrot import (
    mandelbrot_dask,
    mandelbrot_naive,
    mandelbrot_numba_basic,
    mandelbrot_numpy,
    mandelbrot_parallel,
    mandelbrot_pixel,
)


SMALL_GRID = (16, 16, 20, -2.0, 0.5, -1.25, 1.25)


@pytest.mark.parametrize(
    "c_real, c_imag, max_iter, expected",
    [
        (0.0, 0.0, 20, 20),
        (2.0, 0.0, 20, 2),
        (-1.0, 0.0, 20, 20),
    ],
)
def test_mandelbrot_pixel_known_values(c_real, c_imag, max_iter, expected):
    assert mandelbrot_pixel(c_real, c_imag, max_iter) == expected


@pytest.mark.parametrize("implementation", ["numpy", "numba_f64"])
def test_small_grid_matches_naive(implementation):
    expected = np.array(mandelbrot_naive(*SMALL_GRID), dtype=np.int32)

    if implementation == "numpy":
        result = mandelbrot_numpy(*SMALL_GRID)
    else:
        result = mandelbrot_numba_basic(*SMALL_GRID, dtype=np.float64)

    assert np.array_equal(result, expected)


def test_parallel_matches_serial_on_small_grid():
    expected = mandelbrot_numba_basic(*SMALL_GRID, dtype=np.float64)
    result = mandelbrot_parallel(*SMALL_GRID, n_workers=2)

    assert np.array_equal(result, expected)


def test_dask_matches_serial_on_small_grid():
    expected = mandelbrot_numba_basic(*SMALL_GRID, dtype=np.float64)
    result = mandelbrot_dask(*SMALL_GRID, n_chunks=4)

    assert np.array_equal(result, expected)
