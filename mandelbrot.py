import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats
import io
import numpy as np
from numba import njit, prange
import time


def mandelbrot(width, height, max_iter, xmin, xmax, ymin, ymax):
    img = [[0] * width for _ in range(height)]

    for y in range(height):
        im = ymin + (y / (height - 1)) * (ymax - ymin)

        for x in range(width):
            re = xmin + (x / (width - 1)) * (xmax - xmin)
            c = complex(re, im)

            z = 0j
            n = 0

            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
                z = z * z + c
                n += 1

            img[y][x] = n

    return img


def mandelbrot_numpy(width, height, max_iter, xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)

    img = np.zeros(C.shape, dtype=np.int32)

    for n in range(max_iter):
        mask = (Z.real * Z.real + Z.imag * Z.imag) <= 4.0
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        img[mask] = n + 1

    return img


def profile_call(label, func, *args, sort_by="cumulative", top=25, dump_file=None):
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args)
    pr.disable()

    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sort_by)
    stats.print_stats(top)

    print(f"\n=== cProfile: {label} (sorted by {sort_by}) ===")
    print(s.getvalue())

    if dump_file:
        pr.dump_stats(dump_file)

    return result

def time_call(label, func, *args, repeat=5):
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        dt = t1 - t0
        best = dt if best is None else min(best, dt)
    print(f"{label}: best of {repeat} = {best:.6f} s")


@njit(parallel=True, fastmath=True)
def mandelbrot_numba(width, height, max_iter, xmin, xmax, ymin, ymax):
    img = np.empty((height, width), dtype=np.int32)

    for y in prange(height):
        im = ymin + (y / (height - 1)) * (ymax - ymin)
        for x in range(width):
            re = xmin + (x / (width - 1)) * (xmax - xmin)

            zr = 0.0
            zi = 0.0
            cr = re
            ci = im

            n = 0
            while n < max_iter and (zr*zr + zi*zi) <= 4.0:
                zr_new = zr*zr - zi*zi + cr
                zi = 2.0*zr*zi + ci
                zr = zr_new
                n += 1

            img[y, x] = n

    return img


width = 1024
height = 1024
max_iter = 100
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5

naive_data = profile_call(
    "mandelbrot (naive Python)",
    mandelbrot,
    width, height, max_iter, xmin, xmax, ymin, ymax,
    top=100,
)

numpy_data = profile_call(
    "mandelbrot (NumPy)",
    mandelbrot_numpy,
    width, height, max_iter, xmin, xmax, ymin, ymax,
    top=100,
)


mandelbrot_numba(width, height, max_iter, xmin, xmax, ymin, ymax)


numba_data = profile_call(
    "mandelbrot (Numba, warmed up)",
    mandelbrot_numba,
    width, height, max_iter, xmin, xmax, ymin, ymax,
    top=100,
)

#Time on the numba kernel instead of only cProfile
time_call(
    "mandelbrot (Numba, warmed up)",
    mandelbrot_numba,
    width, height, max_iter, xmin, xmax, ymin, ymax,
    repeat=5,
)

plt.figure()
plt.imshow(naive_data, extent=(xmin, xmax, ymin, ymax), origin="lower")
plt.title("Mandelbrot set (naive Python)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.savefig("mandelbrot_naive.png", dpi=150, bbox_inches="tight")

plt.figure()
plt.imshow(numpy_data, extent=(xmin, xmax, ymin, ymax), origin="lower")
plt.title("Mandelbrot set (NumPy)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.savefig("mandelbrot_numpy.png", dpi=150, bbox_inches="tight")

data = mandelbrot_numba(width, height, max_iter, xmin, xmax, ymin, ymax)
plt.imshow(data, extent=(xmin, xmax, ymin, ymax), origin="lower")
plt.title("Mandelbrot set (Numba)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.savefig("mandelbrot_numba.png", dpi=150, bbox_inches="tight")