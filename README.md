# DTAP-MPPI
DTAP-MPPI (Dynamic Terrain-Aware Payload MPPI) is an augmented control algorithm based on the Model-Predictive Path Integral (MPPI) Framework. The augmentations optimize the MPPI control framework for payload transportation robots. These robots are designed to transport various packages from a starting point to a goal region in an environment with dynamic obstacles at variable density. This repository utilizes Numba-CUDA for GPU-accelerated programming in a Python environment. 

## Dependencies
This repository requires the following before usage:
* Python 3.13
* An NVIDIA GPU supported by Numba
* CUDA Toolkit
* uv (Refer to https://docs.astral.sh/uv/getting-started/installation/ for installation)