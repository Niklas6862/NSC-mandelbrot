from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mandelbrot import mandelbrot_pixel


def test_mandelbrot_pixel_known_points():
    max_iter = 20

    # c = 0 stays at z_n = 0 forever, so it should never escape.
    assert mandelbrot_pixel(0.0, 0.0, max_iter) == max_iter

    # c = 2 gives z_1 = 2 and z_2 = 6, so the loop records 2 iterations.
    assert mandelbrot_pixel(2.0, 0.0, max_iter) == 2
