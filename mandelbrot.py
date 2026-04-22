from __future__ import annotations

import numpy as np
import numpy.typing as npt
import cProfile
import pstats
import io
import time
from numba import njit, prange
from multiprocessing.pool import Pool
import multiprocessing as mp

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

EPS32 = np.float64(np.finfo(np.float32).eps)
ChunkArgs = tuple[int, int, int, int, int, float, float, float, float]


@njit(fastmath=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
    """Compute the Mandelbrot escape count for a single complex coordinate.

    Parameters
    ----------
    c_real : float
        Real part of the complex coordinate ``c``.
    c_imag : float
        Imaginary part of the complex coordinate ``c``.
    max_iter : int
        Maximum number of iterations allowed before treating the point as
        non-escaping.

    Returns
    -------
    int
        Number of iterations completed before escape, or ``max_iter`` if the
        point does not escape within the iteration limit.
    """
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
def mandelbrot_chunk(
    row_start: int,
    row_end: int,
    width: int,
    height: int,
    max_iter: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> npt.NDArray[np.int32]:
    """Compute a contiguous band of Mandelbrot rows.

    Parameters
    ----------
    row_start : int
        First row index included in the chunk.
    row_end : int
        Row index just past the end of the chunk.
    width : int
        Number of columns in the output grid.
    height : int
        Total grid height, used to map row indices to imaginary coordinates.
    max_iter : int
        Maximum number of Mandelbrot iterations per pixel.
    xmin : float
        Lower bound of the real-axis interval.
    xmax : float
        Upper bound of the real-axis interval.
    ymin : float
        Lower bound of the imaginary-axis interval.
    ymax : float
        Upper bound of the imaginary-axis interval.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``int32`` array containing escape counts for the rows
        in ``[row_start, row_end)``.
    """
    img = np.empty((row_end - row_start, width), dtype=np.int32)

    for y in range(row_start, row_end):
        im = ymin + (y / (height - 1)) * (ymax - ymin)

        for x in range(width):
            re = xmin + (x / (width - 1)) * (xmax - xmin)
            img[y - row_start, x] = mandelbrot_pixel(re, im, max_iter)

    return img


def mandelbrot_serial(
    width: int,
    height: int,
    max_iter: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> npt.NDArray[np.int32]:
    """Compute a full Mandelbrot image on one process.

    Parameters
    ----------
    width : int
        Number of sample points along the real axis.
    height : int
        Number of sample points along the imaginary axis.
    max_iter : int
        Maximum number of Mandelbrot iterations per pixel.
    xmin : float
        Lower bound of the real-axis interval.
    xmax : float
        Upper bound of the real-axis interval.
    ymin : float
        Lower bound of the imaginary-axis interval.
    ymax : float
        Upper bound of the imaginary-axis interval.

    Returns
    -------
    numpy.ndarray
        Full ``height x width`` escape-count image as an ``int32`` array.
    """
    return mandelbrot_chunk(0, height, width, height, max_iter, xmin, xmax, ymin, ymax)


def _worker_func(item: ChunkArgs) -> npt.NDArray[np.int32]:
    row_start, row_end, width, height, max_iter, xmin, xmax, ymin, ymax = item
    return mandelbrot_chunk(row_start, row_end, width, height, max_iter, xmin, xmax, ymin, ymax)


def build_chunks(
    width: int,
    height: int,
    max_iter: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n_workers: int,
) -> list[ChunkArgs]:
    """Split a Mandelbrot grid into row chunks for parallel execution.

    Parameters
    ----------
    width : int
        Number of sample points along the real axis.
    height : int
        Number of sample points along the imaginary axis.
    max_iter : int
        Maximum number of Mandelbrot iterations per pixel.
    xmin : float
        Lower bound of the real-axis interval.
    xmax : float
        Upper bound of the real-axis interval.
    ymin : float
        Lower bound of the imaginary-axis interval.
    ymax : float
        Upper bound of the imaginary-axis interval.
    n_workers : int
        Number of worker partitions to target when splitting rows.

    Returns
    -------
    list of tuple
        Chunk argument tuples that can be passed directly to worker
        functions for serial or multiprocessing execution.
    """
    chunk_size = (height + n_workers - 1) // n_workers
    chunks = []

    for row_start in range(0, height, chunk_size):
        row_end = min(row_start + chunk_size, height)
        chunks.append((row_start, row_end, width, height, max_iter, xmin, xmax, ymin, ymax))

    return chunks


def mandelbrot_parallel(
    width: int,
    height: int,
    max_iter: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n_workers: int,
    pool: Pool | None = None,
) -> npt.NDArray[np.int32]:
    """Compute a Mandelbrot image with a multiprocessing pool.

    Parameters
    ----------
    width : int
        Number of sample points along the real axis.
    height : int
        Number of sample points along the imaginary axis.
    max_iter : int
        Maximum number of Mandelbrot iterations per pixel.
    xmin : float
        Lower bound of the real-axis interval.
    xmax : float
        Upper bound of the real-axis interval.
    ymin : float
        Lower bound of the imaginary-axis interval.
    ymax : float
        Upper bound of the imaginary-axis interval.
    n_workers : int
        Number of worker processes to use when creating chunk tasks.
    pool : multiprocessing.pool.Pool or None, optional
        Existing pool to reuse. If ``None``, the function creates and owns a
        new pool for the duration of the call.

    Returns
    -------
    numpy.ndarray
        Full ``height x width`` escape-count image assembled from the worker
        chunks.
    """
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


def mandelbrot_trajectory_divergence(
    width, height, max_iter, xmin, xmax, ymin, ymax, tau=0.01
):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    c64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    c32 = c64.astype(np.complex64)

    z64 = np.zeros_like(c64)
    z32 = np.zeros_like(c32)

    diverge = np.full((height, width), max_iter, dtype=np.int32)
    active = np.ones((height, width), dtype=bool)

    for k in range(max_iter):
        if not active.any():
            break

        z32[active] = z32[active] * z32[active] + c32[active]
        z64[active] = z64[active] * z64[active] + c64[active]

        diff = (
            np.abs(z32.real.astype(np.float64) - z64.real)
            + np.abs(z32.imag.astype(np.float64) - z64.imag)
        )
        newly_diverged = active & (diff > tau)

        diverge[newly_diverged] = k + 1
        active[newly_diverged] = False

    return diverge


@njit(parallel=True, fastmath=True)
def mandelbrot_sensitivity_map_kernel(
    width, height, max_iter, xmin, xmax, ymin, ymax
):
    kappa = np.empty((height, width), dtype=np.float64)
    n_base = np.empty((height, width), dtype=np.float64)
    n_perturb = np.empty((height, width), dtype=np.float64)

    xmin_f = np.float64(xmin)
    xmax_f = np.float64(xmax)
    ymin_f = np.float64(ymin)
    ymax_f = np.float64(ymax)

    for y in prange(height):
        im = ymin_f + (np.float64(y) / (height - 1)) * (ymax_f - ymin_f)

        for x in range(width):
            re = xmin_f + (np.float64(x) / (width - 1)) * (xmax_f - xmin_f)
            base = mandelbrot_pixel(re, im, max_iter)

            delta = EPS32 * np.sqrt(re * re + im * im)
            if delta < 1e-10:
                delta = 1e-10

            perturb = mandelbrot_pixel(re + delta, im, max_iter)

            n_base[y, x] = base
            n_perturb[y, x] = perturb

            if base > 0:
                kappa[y, x] = np.abs(perturb - base) / (EPS32 * base)
            else:
                kappa[y, x] = np.nan

    return kappa, n_base, n_perturb


def mandelbrot_sensitivity_map(width, height, max_iter, xmin, xmax, ymin, ymax):
    """Approximate a per-pixel Mandelbrot sensitivity map over a 2D region.

    Parameters
    ----------
    width : int
        Number of sample points along the real axis.
    height : int
        Number of sample points along the imaginary axis.
    max_iter : int
        Maximum number of Mandelbrot iterations used for each pixel.
    xmin : float
        Lower bound of the real-axis interval.
    xmax : float
        Upper bound of the real-axis interval.
    ymin : float
        Lower bound of the imaginary-axis interval.
    ymax : float
        Upper bound of the imaginary-axis interval.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple ``(kappa, n_base, n_perturb)`` where ``kappa`` is the
        condition-number approximation for each pixel, ``n_base`` is the
        unperturbed escape-count map, and ``n_perturb`` is the escape-count
        map after perturbing the real part by ``eps32 * |c|``.

    Examples
    --------
    >>> kappa, n_base, n_perturb = mandelbrot_sensitivity_map(
    ...     64, 64, 100, -0.7530, -0.7490, 0.0990, 0.1030
    ... )
    >>> kappa.shape
    (64, 64)
    """
    return mandelbrot_sensitivity_map_kernel(
        width, height, max_iter, xmin, xmax, ymin, ymax
    )

# ------ L04 MP2 ------
