import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats
import io
import time
from numba import njit, prange


def time_call(label, func, *args):
    t0 = time.perf_counter()
    func(*args)
    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"{label}: {dt:.6f} s")


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


def line_profile_call(label, func, *args, extra_funcs=()):
    try:
        from line_profiler import LineProfiler
    except ImportError as e:
        raise SystemExit(
            "Missing dependency 'line_profiler'. Install with:\n"
            "  python -m pip install line_profiler"
        ) from e

    lp = LineProfiler()
    lp.add_function(func)

    for f in extra_funcs:
        lp.add_function(f)

    result = lp.runcall(func, *args)

    print(f"\n=== line_profiler: {label} ===")
    lp.print_stats()

    return result


# implementations
def mandelbrot_naive(width, height, max_iter, xmin, xmax, ymin, ymax):
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


@njit(fastmath=True)
def mandelbrot_numba_basic(width, height, max_iter, xmin, xmax, ymin, ymax, dtype):
    img = np.empty((height, width), dtype=np.int32)
    
    xmin_f = dtype(xmin)
    xmax_f = dtype(xmax)
    ymin_f = dtype(ymin)
    ymax_f = dtype(ymax)

    for y in range(height):
        im = ymin_f + (dtype(y) / (height - 1)) * (ymax_f - ymin_f)

        for x in range(width):
            re = xmin_f + (dtype(x) / (width - 1)) * (xmax_f - xmin_f)

            zr = dtype(0.0)
            zi = dtype(0.0)
            cr = re
            ci = im

            n = 0
            while n < max_iter and (zr * zr + zi * zi) <= dtype(4.0):
                zr_new = zr * zr - zi * zi + cr
                zi = dtype(2.0) * zr * zi + ci
                zr = zr_new
                n += 1

            img[y, x] = n

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
            while n < max_iter and (zr * zr + zi * zi) <= 4.0:
                zr_new = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = zr_new
                n += 1

            img[y, x] = n

    return img


# Parameters
width = 1024
height = 1024
max_iter = 100
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5


# Profiling & Timing

# Line-profile naive
naive_data = line_profile_call(
    "mandelbrot (naive Python)",
    mandelbrot_naive,
    width, height, max_iter, xmin, xmax, ymin, ymax
)

time_call(
    "mandelbrot (naive)",
    mandelbrot_naive,
    width, height, max_iter, xmin, xmax, ymin, ymax
)

# NumPy profiling
numpy_data = profile_call(
    "mandelbrot (NumPy)",
    mandelbrot_numpy,
    width, height, max_iter, xmin, xmax, ymin, ymax,
    top=100
)

time_call(
    "mandelbrot (NumPy)",
    mandelbrot_numpy,
    width, height, max_iter, xmin, xmax, ymin, ymax
)

# Numba naive-loop and profiling (warmed up)
mandelbrot_numba_basic(width, height, max_iter, xmin, xmax, ymin, ymax, np.float64)

numba_basic = profile_call(
    "mandelbrot (Numba @njit naive loop, warmed up)",
    mandelbrot_numba_basic,
    width, height, max_iter, xmin, xmax, ymin, ymax, np.float64,
    top=100
)

time_call(
    "mandelbrot (Numba @njit naive loop, warmed up)",
    mandelbrot_numba_basic,
    width, height, max_iter, xmin, xmax, ymin, ymax, np.float64
)

# Numba parallel and profiling (warmed up)
mandelbrot_numba(width, height, max_iter, xmin, xmax, ymin, ymax)

numba_data = profile_call(
    "mandelbrot (Numba, warmed up)",
    mandelbrot_numba,
    width, height, max_iter, xmin, xmax, ymin, ymax,
    top=100
)

time_call(
    "mandelbrot (Numba, warmed up)",
    mandelbrot_numba,
    width, height, max_iter, xmin, xmax, ymin, ymax
)

# Plotting
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

plt.figure()
plt.imshow(numba_basic, extent=(xmin, xmax, ymin, ymax), origin="lower")
plt.title("Mandelbrot set (Numba basic)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.savefig("mandelbrot_numba_basic.png", dpi=150, bbox_inches="tight")

plt.figure()
plt.imshow(numba_data, extent=(xmin, xmax, ymin, ymax), origin="lower")
plt.title("Mandelbrot set (Numba)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.savefig("mandelbrot_numba_parallel.png", dpi=150, bbox_inches="tight")
