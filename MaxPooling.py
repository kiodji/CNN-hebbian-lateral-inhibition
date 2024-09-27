import numpy as np
from numba import cuda

class MaxPooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        depth, height, width = input.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1

        d_input = cuda.to_device(input)
        d_output = cuda.device_array((depth, output_height, output_width), dtype=input.dtype)

        threads_per_block = (8, 8, 1)
        blocks_per_grid = (
            (output_height + threads_per_block[0] - 1) // threads_per_block[0],
            (output_width + threads_per_block[1] - 1) // threads_per_block[1],
            depth
        )

        max_pooling_kernel[blocks_per_grid, threads_per_block](d_input, d_output, self.pool_size, self.stride)
        return d_output.copy_to_host()

@cuda.jit
def max_pooling_kernel(input, output, pool_size, stride):
    d, x, y = cuda.grid(3)
    if d < input.shape[0] and x < output.shape[1] and y < output.shape[2]:
        max_val = -99999
        for i in range(pool_size):
            for j in range(pool_size):
                xi = x * stride + i
                yj = y * stride + j
                if xi < input.shape[1] and yj < input.shape[2]:
                    val = input[d, xi, yj]
                    if val > max_val:
                        max_val = val
        output[d, x, y] = max_val