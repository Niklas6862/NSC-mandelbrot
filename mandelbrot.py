import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats
import io
import time
from numba import float64, njit, prange

# Parameters
width = 1024
height = 1024
max_iter = 100
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5

# ------ L01 to L03 MP1 ------

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

    print(f"\n cProfile: {label} (sorted by {sort_by}) \n")
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

    print(f"\n line_profiler: {label} \n")
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


@njit(fastmath=True)
def mandelbrot_numba_basic_f64(width, height, max_iter, xmin, xmax, ymin, ymax):
    img = np.empty((height, width), dtype=np.int32)

    xmin_f = np.float64(xmin)
    xmax_f = np.float64(xmax)
    ymin_f = np.float64(ymin)
    ymax_f = np.float64(ymax)

    for y in range(height):
        im = ymin_f + (np.float64(y) / (height - 1)) * (ymax_f - ymin_f)

        for x in range(width):
            re = xmin_f + (np.float64(x) / (width - 1)) * (xmax_f - xmin_f)

            zr = np.float64(0.0)
            zi = np.float64(0.0)
            cr = re
            ci = im

            n = 0
            while n < max_iter and (zr * zr + zi * zi) <= np.float64(4.0):
                zr_new = zr * zr - zi * zi + cr
                zi = np.float64(2.0) * zr * zi + ci
                zr = zr_new
                n += 1

            img[y, x] = n

    return img


@njit(fastmath=True)
def mandelbrot_numba_basic_f32(width, height, max_iter, xmin, xmax, ymin, ymax):
    img = np.empty((height, width), dtype=np.int32)

    xmin_f = np.float32(xmin)
    xmax_f = np.float32(xmax)
    ymin_f = np.float32(ymin)
    ymax_f = np.float32(ymax)

    for y in range(height):
        im = ymin_f + (np.float32(y) / (height - 1)) * (ymax_f - ymin_f)

        for x in range(width):
            re = xmin_f + (np.float32(x) / (width - 1)) * (xmax_f - xmin_f)

            zr = np.float32(0.0)
            zi = np.float32(0.0)
            cr = re
            ci = im

            n = 0
            while n < max_iter and (zr * zr + zi * zi) <= np.float32(4.0):
                zr_new = zr * zr - zi * zi + cr
                zi = np.float32(2.0) * zr * zi + ci
                zr = zr_new
                n += 1

            img[y, x] = n

    return img

@njit(fastmath=True)
def mandelbrot_numba_basic_f16(width, height, max_iter, xmin, xmax, ymin, ymax):
    img = np.empty((height, width), dtype=np.int32)

    xmin_f = np.float16(xmin)
    xmax_f = np.float16(xmax)
    ymin_f = np.float16(ymin)
    ymax_f = np.float16(ymax)

    for y in range(height):
        im = ymin_f + (np.float16(y) / (height - 1)) * (ymax_f - ymin_f)

        for x in range(width):
            re = xmin_f + (np.float16(x) / (width - 1)) * (xmax_f - xmin_f)

            zr = np.float16(0.0)
            zi = np.float16(0.0)
            cr = re
            ci = im

            n = 0
            while n < max_iter and (zr * zr + zi * zi) <= np.float16(4.0):
                zr_new = zr * zr - zi * zi + cr
                zi = np.float16(2.0) * zr * zi + ci
                zr = zr_new
                n += 1

            img[y, x] = n

    return img


def mandelbrot_numba_basic(width, height, max_iter, xmin, xmax, ymin, ymax, dtype=np.float64):
    if dtype == np.float32:
        return mandelbrot_numba_basic_f32(width, height, max_iter, xmin, xmax, ymin, ymax)
    elif dtype == np.float16:
        return mandelbrot_numba_basic_f16(width, height, max_iter, xmin, xmax, ymin, ymax)
    return mandelbrot_numba_basic_f64(width, height, max_iter, xmin, xmax, ymin, ymax)

# ------ L01 to L03 MP1 ------
# ------ L04 MP2 ------

import numpy as np
from numba import njit
import multiprocessing as mp


@njit(fastmath=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    zr = 0.0
    zi = 0.0
    n = 0

    while n < max_iter and (zr * zr + zi * zi) <= 4.0:
        zr_new = zr * zr - zi * zi + c_real
        zi = 2.0 * zr * zi + c_imag
        zr = zr_new
        n += 1

    return n


@njit(fastmath=True)
def mandelbrot_chunk(row_start, row_end, width, height, max_iter, xmin, xmax, ymin, ymax):
    img = np.empty((row_end - row_start, width), dtype=np.int32)

    for y in range(row_start, row_end):
        im = ymin + (y / (height - 1)) * (ymax - ymin)

        for x in range(width):
            re = xmin + (x / (width - 1)) * (xmax - xmin)
            img[y - row_start, x] = mandelbrot_pixel(re, im, max_iter)

    return img


def mandelbrot_serial(width, height, max_iter, xmin, xmax, ymin, ymax):
    return mandelbrot_chunk(0, height, width, height, max_iter, xmin, xmax, ymin, ymax)


def _worker_func(item):
    row_start, row_end, width, height, max_iter, xmin, xmax, ymin, ymax = item
    return mandelbrot_chunk(row_start, row_end, width, height, max_iter, xmin, xmax, ymin, ymax)


def build_chunks(width, height, max_iter, xmin, xmax, ymin, ymax, n_workers):
    chunk_size = (height + n_workers - 1) // n_workers
    chunks = []

    for row_start in range(0, height, chunk_size):
        row_end = min(row_start + chunk_size, height)
        chunks.append((row_start, row_end, width, height, max_iter, xmin, xmax, ymin, ymax))

    return chunks


def mandelbrot_parallel(width, height, max_iter, xmin, xmax, ymin, ymax, n_workers, pool=None):
    chunks = build_chunks(width, height, max_iter, xmin, xmax, ymin, ymax, n_workers)
    owns_pool = pool is None

    if owns_pool:
        pool = mp.Pool(processes=n_workers)

    try:
        parts = pool.map(_worker_func, chunks)
    finally:
        if owns_pool:
            pool.close()
            pool.join()

    return np.vstack(parts)


def mandelbrot_dask(width, height, max_iter, xmin, xmax, ymin, ymax, n_chunks=32):
    from dask import delayed, compute

    chunk_args = build_chunks(width, height, max_iter, xmin, xmax, ymin, ymax, n_chunks)
    tasks = [delayed(mandelbrot_chunk)(*args) for args in chunk_args]

    parts = compute(*tasks)
    return np.vstack(parts)

# ------ L04 MP2 ------
