
# jax by google

## Installation
- `conda install cudnn` -- This automatically installs cuda too, select the cuda-nvcc version matching with cuda version
- `conda install -c nvidia cuda-nvcc=<version>`

## References
Library [link](https://github.com/google/jax)<br>

## FAQ

- segfault due to cuda version mismatch
  - https://github.com/google/jax/issues/5301
- OOM/determine cudnn convolution algorithm error
  - https://github.com/google/jax/issues/8746
  - solution
  ```sh
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

  ```

