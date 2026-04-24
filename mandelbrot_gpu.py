from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np


KERNEL_SRC_F32 = """
__kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col >= N || row >= N) {
        return;
    }

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float zr = 0.0f;
    float zi = 0.0f;
    int count = 0;

    while (count < max_iter && zr * zr + zi * zi <= 4.0f) {
        float tmp = zr * zr - zi * zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}
"""


KERNEL_SRC_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col >= N || row >= N) {
        return;
    }

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double zr = 0.0;
    double zi = 0.0;
    int count = 0;

    while (count < max_iter && zr * zr + zi * zi <= 4.0) {
        double tmp = zr * zr - zi * zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}
"""


MAX_ITER = 200
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25
BENCHMARK_SIZES = [1024, 2048]


def build_kernel(cl, ctx, kernel_src: str):
    prog = cl.Program(ctx, kernel_src).build()
    return prog, cl.Kernel(prog, "mandelbrot")


def find_cpu_fp64_device(cl):
    for platform in cl.get_platforms():
        try:
            devices = platform.get_devices(device_type=cl.device_type.CPU)
        except Exception:
            continue

        for dev in devices:
            if "cl_khr_fp64" in getattr(dev, "extensions", ""):
                return dev

    return None


def run_mandelbrot(
    cl,
    queue,
    kernel,
    n: int,
    max_iter: int,
    x_min,
    x_max,
    y_min,
    y_max,
    scalar_dtype,
):
    image = np.zeros((n, n), dtype=np.int32)
    image_dev = cl.Buffer(queue.context, cl.mem_flags.WRITE_ONLY, image.nbytes)

    warm_n = min(64, n)
    kernel(
        queue,
        (warm_n, warm_n),
        None,
        image_dev,
        scalar_dtype(x_min),
        scalar_dtype(x_max),
        scalar_dtype(y_min),
        scalar_dtype(y_max),
        np.int32(warm_n),
        np.int32(max_iter),
    )
    queue.finish()

    t0 = time.perf_counter()
    kernel(
        queue,
        (n, n),
        None,
        image_dev,
        scalar_dtype(x_min),
        scalar_dtype(x_max),
        scalar_dtype(y_min),
        scalar_dtype(y_max),
        np.int32(n),
        np.int32(max_iter),
    )
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()

    return image, elapsed


def save_image(image: np.ndarray, filename: str) -> None:
    plt.figure()
    plt.imshow(image, cmap="hot", origin="lower")
    plt.axis("off")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    try:
        import pyopencl as cl
    except ImportError as e:
        raise SystemExit(
            "PyOpenCL is not installed in this environment. "
            "Install it first, then rerun mandelbrot_gpu.py."
        ) from e

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    dev = ctx.devices[0]

    _, kernel_f32 = build_kernel(cl, ctx, KERNEL_SRC_F32)
    has_fp64 = "cl_khr_fp64" in getattr(dev, "extensions", "")

    kernel_f64 = None
    queue_f64 = queue
    dev_f64 = dev
    f64_mode = "native"

    if has_fp64:
        try:
            _, kernel_f64 = build_kernel(cl, ctx, KERNEL_SRC_F64)
        except Exception:
            print("Could not build the fp64 kernel on this device; skipping float64 benchmark.")
    else:
        print("No native fp64 detected on this GPU.")
        cpu_dev = find_cpu_fp64_device(cl)

        if cpu_dev is not None:
            print(f"Using CPU OpenCL device for emulated/software fp64: {cpu_dev.name}")
            ctx_f64 = cl.Context([cpu_dev])
            queue_f64 = cl.CommandQueue(ctx_f64)
            dev_f64 = cpu_dev
            f64_mode = "emulated"
            try:
                _, kernel_f64 = build_kernel(cl, ctx_f64, KERNEL_SRC_F64)
            except Exception:
                print("Could not build the fp64 kernel on the CPU OpenCL device; skipping float64 benchmark.")
                kernel_f64 = None
        else:
            print("No CPU OpenCL device with fp64 support was found; skipping float64 benchmark.")

    timings_f32: dict[int, float] = {}
    timings_f64: dict[int, float] = {}

    for n in BENCHMARK_SIZES:
        image_f32, elapsed_f32 = run_mandelbrot(
            cl,
            queue,
            kernel_f32,
            n,
            MAX_ITER,
            X_MIN,
            X_MAX,
            Y_MIN,
            Y_MAX,
            np.float32,
        )
        timings_f32[n] = elapsed_f32
        print(f"GPU f32 {n}x{n}: {elapsed_f32 * 1e3:.1f} ms")

        if n == BENCHMARK_SIZES[0]:
            save_image(image_f32, "mandelbrot_gpu_f32.png")

        if kernel_f64 is not None:
            image_f64, elapsed_f64 = run_mandelbrot(
                cl,
                queue_f64,
                kernel_f64,
                n,
                MAX_ITER,
                X_MIN,
                X_MAX,
                Y_MIN,
                Y_MAX,
                np.float64,
            )
            timings_f64[n] = elapsed_f64
            label_f64 = "GPU f64" if f64_mode == "native" else "CPU-emulated f64"
            print(f"{label_f64} {n}x{n}: {elapsed_f64 * 1e3:.1f} ms")
            ratio_label = "f32/f64 speed ratio"
            if f64_mode != "native":
                ratio_label = "f32/native vs f64/emulated ratio"
            print(f"{ratio_label} at {n}: {elapsed_f64 / elapsed_f32:.2f}x slower")

            if n == BENCHMARK_SIZES[0]:
                save_image(image_f64, "mandelbrot_gpu_f64.png")
                max_abs_diff = np.max(np.abs(image_f32.astype(np.int32) - image_f64.astype(np.int32)))
                print(f"max |f32 - f64| at {n}x{n}: {max_abs_diff}")


if __name__ == "__main__":
    main()
