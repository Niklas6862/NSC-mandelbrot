from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np


KERNEL_SRC = """
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

    float c_real = x_min + col * (x_max - x_min) / (float)(N - 1);
    float c_imag = y_min + row * (y_max - y_min) / (float)(N - 1);

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


N = 1024
MAX_ITER = 200
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25


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
    prog = cl.Program(ctx, KERNEL_SRC).build()

    image = np.zeros((N, N), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

    # Warm up with a small launch so build and first-run overhead are paid once.
    prog.mandelbrot(
        queue,
        (64, 64),
        None,
        image_dev,
        np.float32(X_MIN),
        np.float32(X_MAX),
        np.float32(Y_MIN),
        np.float32(Y_MAX),
        np.int32(64),
        np.int32(MAX_ITER),
    )
    queue.finish()

    t0 = time.perf_counter()
    prog.mandelbrot(
        queue,
        (N, N),
        None,
        image_dev,
        np.float32(X_MIN),
        np.float32(X_MAX),
        np.float32(Y_MIN),
        np.float32(Y_MAX),
        np.int32(N),
        np.int32(MAX_ITER),
    )
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()

    print(f"GPU {N}x{N}: {elapsed * 1e3:.1f} ms")

    plt.imshow(image, cmap="hot", origin="lower")
    plt.axis("off")
    plt.savefig("mandelbrot_gpu.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
