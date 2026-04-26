# NSC-mandelbrot

Course repository for Mandelbrot implementations and performance experiments in Python, Numba, multiprocessing, Dask, and OpenCL.

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 2. Install the core packages

```bash
python -m pip install --upgrade pip
python -m pip install numpy matplotlib numba pytest dask distributed
```

These packages are enough for:

- `mandelbrot.py`
- `MP1.ipynb`
- `MP2.ipynb`
- `MP3.ipynb`
- the test suite in `tests/`

### 3. Optional GPU setup

For the OpenCL part in `mandelbrot_gpu.py`, install PyOpenCL:

```bash
python -m pip install pyopencl
```

This step is optional. The GPU milestone only works on systems where OpenCL and PyOpenCL are available.

## Running the code

Run the main Python implementations:

```bash
python mandelbrot.py
python mandelbrot_dask.py
python mandelbrot_gpu.py
```

Open the notebooks with Jupyter:

```bash
python -m pip install notebook
jupyter notebook
```

## Running tests

Run the test suite with:

```bash
pytest -v
```

If you also want coverage and have `pytest-cov` installed:

```bash
python -m pip install pytest-cov
pytest --cov=. -v
```

## Notes

- `mandelbrot_gpu.py` uses OpenCL float32 by default.
- On Apple GPUs, OpenCL float64 may be unavailable, so the float64 benchmark can be skipped or fall back to a CPU OpenCL device if one exists.
- Dask cluster code in `MP2.ipynb` assumes a separate Strato setup and is not required for basic local setup.
