from dask.distributed import Client
from mandelbrot import mandelbrot_serial, mandelbrot_dask, mandelbrot_chunk
import matplotlib.pyplot as plt
import numpy as np
import os
import statistics
import time


scheduler_address = os.environ.get("DASK_SCHEDULER_ADDRESS", "tcp://10.92.1.110:8786")
width = 4096
height = 4096
max_iter = 100
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
n_chunk_values = [4, 8, 16, 32, 64, 128, 256]


def version_info():
    import dask
    import distributed
    import numpy
    import socket
    import sys

    return {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "dask": dask.__version__,
        "distributed": distributed.__version__,
    }


def warmup_worker():
    return mandelbrot_chunk(0, 8, 8, 8, 10, xmin, xmax, ymin, ymax)


def time_serial_3x(width, height, max_iter, xmin, xmax, ymin, ymax):
    times = []
    last_img = None

    for _ in range(3):
        t0 = time.perf_counter()
        img = mandelbrot_serial(width, height, max_iter, xmin, xmax, ymin, ymax)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_img = img

    return statistics.median(times), last_img


def time_dask_3x(width, height, max_iter, xmin, xmax, ymin, ymax, n_chunks):
    times = []
    last_img = None

    for _ in range(3):
        t0 = time.perf_counter()
        img = mandelbrot_dask(width, height, max_iter, xmin, xmax, ymin, ymax, n_chunks=n_chunks)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_img = img

    return statistics.median(times), last_img


if __name__ == "__main__":
    try:
        client = Client(scheduler_address, timeout="30s")
    except OSError as e:
        raise SystemExit(
            f"Could not connect to Dask scheduler at {scheduler_address}.\n"
            "Make sure `dask scheduler` is running on the head node,\n"
            "workers are started with the same scheduler IP,\n"
            "and port 8786 is open in the Strato security group."
        ) from e

    print(client)
    print(client.run(version_info))

    warm_args = (256, 256, max_iter, xmin, xmax, ymin, ymax)
    mandelbrot_serial(*warm_args)
    client.run(warmup_worker)

    worker_processes = len(client.scheduler_info()["workers"])
    print(f"Connected worker processes: {worker_processes}")

    t_serial, serial_img = time_serial_3x(width, height, max_iter, xmin, xmax, ymin, ymax)
    print(f"Serial Numba baseline: {t_serial:.3f} s")
    print()

    chunk_results = []
    t_1x = None

    print("n_chunks | time (s) | vs 1x | speedup | LIF")

    for n_chunks in n_chunk_values:
        t_dask, dask_img = time_dask_3x(width, height, max_iter, xmin, xmax, ymin, ymax, n_chunks)
        assert np.array_equal(serial_img, dask_img)

        if t_1x is None:
            t_1x = t_dask

        vs_1x = t_dask / t_1x
        speedup = t_1x / t_dask
        lif = worker_processes * t_dask / t_1x - 1

        chunk_results.append((n_chunks, t_dask, vs_1x, speedup, lif))
        print(f"{n_chunks:8d} | {t_dask:8.3f} | {vs_1x:5.2f} | {speedup:7.2f} | {lif:5.2f}")

    n_chunks_optimal, t_min, _, _, lif_min = min(chunk_results, key=lambda item: item[1])
    print()
    print(f"n_chunks_optimal = {n_chunks_optimal}")
    print(f"t_min = {t_min:.3f} s")
    print(f"LIF_min = {lif_min:.3f}")
    print(f"speedup vs serial = {t_serial / t_min:.2f}x")

    plt.figure(figsize=(7, 4))
    plt.plot([row[0] for row in chunk_results], [row[1] for row in chunk_results], marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("n_chunks")
    plt.ylabel("time (s)")
    plt.title("Strato Dask chunk sweep")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("strato_chunk_sweep.png", dpi=200)
    plt.show()

    print()
    print("For worker scaling:")
    print("1. Start with one worker VM connected and run this script.")
    print("2. Add another worker VM and run this script again.")
    print("3. Record worker processes, wall time, and speedup vs serial each time.")

    client.close()
