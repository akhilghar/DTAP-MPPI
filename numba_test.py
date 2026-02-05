from numba import vectorize, cuda
import numpy as np

# Define a CUDA vectorized function
@vectorize(['float32(float32, float32)'], target='cuda')
def add_vectors_cuda(a, b):
    return a + b

# Create large arrays in NumPy (host memory)
N = 100000
A = np.ones(N, dtype=np.float32)
B = np.ones(N, dtype=np.float32)

# Numba automatically manages memory transfer and kernel execution
C = add_vectors_cuda(A, B)
print(C)