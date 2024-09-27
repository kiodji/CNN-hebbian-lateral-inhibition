from numba import cuda, jit
import numpy as np
import math

class Layer:
    def __init__(self, filters_num, depth=1, rs=3, learning_rate=0.00001, stride=1):
        self.stride = stride
        self.kernel_size = rs
        self.filters_num = filters_num
        self.depth = depth
        self.pool_size = 2
        self.pool_stride = 2
        self.learning_rate = learning_rate
        self.filters = np.random.uniform(0, 1, (filters_num, depth, rs, rs)).astype(np.float32)
        self.last_output = None
        self.pooled_output_height = 0
        self.pooled_output_width = 0

    def forward(self, input):
        padding_size = self.kernel_size // 2
        input_height, input_width = input[0].shape
        output_height = (input_height + 2 * padding_size - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * padding_size - self.kernel_size) // self.stride + 1

        output = np.zeros((self.filters_num, output_height, output_width), dtype=np.float32)
        inhibition = np.zeros_like(output)

        convolution_kernel(input, self.filters, output, self.kernel_size, self.depth, output_height, output_width, self.stride, padding_size)
        lateral_inhibition(output, inhibition)
        hebbian_weights_update(input, output, inhibition, self.filters, output_height, output_width, self.kernel_size, self.depth, padding_size, self.learning_rate, self.stride)
        normalize_weights_l2(self.filters, self.kernel_size, self.depth, self.filters_num)

        pooled_output_height = (output_height - self.pool_size) // self.pool_stride + 1
        pooled_output_width = (output_width - self.pool_size) // self.pool_stride + 1
        self.pooled_output_height = pooled_output_height
        self.pooled_output_width = pooled_output_width
        pooled_output = np.zeros((self.filters_num, pooled_output_height, pooled_output_width), dtype=np.float32)
        max_pooling_kernel(output, pooled_output, self.pool_size, self.pool_stride)

        self.last_output = pooled_output
        # self.last_output = output
        return self.last_output

@jit(nopython=True)
def convolution_kernel(input, filters, output, kernel_size, channels, height, width, stride, padding_size):
    for f in range(output.shape[0]):
        for x in range(height):
            for y in range(width):
                sum_val = 0.0
                for d in range(channels):
                    for m in range(kernel_size):
                        for n in range(kernel_size):
                            i = x * stride + m - padding_size
                            j = y * stride + n - padding_size
                            if 0 <= i < input.shape[1] and 0 <= j < input.shape[2]:
                                sum_val += input[d, i, j] * filters[f, d, m, n]
                output[f, x, y] = max(0, sum_val / (kernel_size * kernel_size))  # ReLU activation with output normalization
    
# hebbian theory: suggesting that neurons that fire together - wire together.
# Also this theory or formula replicates the biological STDP (when the number of input spikes are more than output spikes,
# the synaptic strength increases. When the number of input spikes are less than output spikes, the synaptic strength decreases
# in our case the number of spikes is a pixel value from 0-255 or 0-1 
@jit(nopython=True)
def hebbian_weights_update(last_input, last_output, inhibition, filters, height, width, kernel_size, channels, padding_size, rate, stride):

    for f in range(last_output.shape[0]):
        for x in range(height):
            for y in range(width):
                output_val = last_output[f, x, y]
                inhibit = inhibition[f, x, y]
                for d in range(channels):
                    for m in range(kernel_size):
                        for n in range(kernel_size):
                            i = x * stride + m - padding_size
                            j = y * stride + n - padding_size
                            if 0 <= i < last_input.shape[1] and 0 <= j < last_input.shape[2]:
                                input_val = last_input[d, i, j]
                                delta_weight = rate * (input_val * output_val) * inhibit #hebbian formula with lateral inhibition
                                filters[f, d, m, n] += delta_weight
@jit(nopython=True)
def lateral_inhibition(output, inhibition):
    for x in range(output.shape[1]):
        for y in range(output.shape[2]):
            max_activation = -1
            max_idx = -1
            for f in range(output.shape[0]):
                activation = output[f, x, y]
                if activation > max_activation:
                    max_activation = activation
                    max_idx = f

            for f in range(output.shape[0]):
                if f == max_idx:
                    inhibition[f, x, y] = 1
                else:
                    inhibition[f, x, y] = 0.01
                    output[f, x, y] *= inhibition[f, x, y]
                
@jit(nopython=True)
def max_pooling_kernel(output, pooled_output, pool_size, stride):
    for f in range(output.shape[0]):
        for x in range(pooled_output.shape[1]):
            for y in range(pooled_output.shape[2]):
                max_value = -1
                for i in range(pool_size):
                    for j in range(pool_size):
                        ni = x * stride + i
                        nj = y * stride + j
                        if ni < output.shape[1] and nj < output.shape[2]:
                            if output[f, ni, nj] > max_value:
                                max_value = output[f, ni, nj]
                pooled_output[f, x, y] = max_value

# @jit(nopython=True)
# def max_pooling_kernel(output, pooled_output, pool_size, stride):
#     h, w = output.shape[1], output.shape[2]

#     for i in range(0, h, stride):
#         for j in range(0, w, stride):
#             pooled_output[:, i // pool_size, j // pool_size] = np.max(output[:, i:i + pool_size, j:j + pool_size], axis=(1, 2))

@jit(nopython=True)
def normalize_weights_l2(weights, kernel_size, depth, filters_num):
    for f in range(filters_num):
        for d in range(depth):
            norm = 0.0
            for i in range(kernel_size):
                for j in range(kernel_size):
                    norm += weights[f, d, i, j] ** 2
            norm = math.sqrt(norm) * 0.7
            if norm > 0:
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        weights[f, d, i, j] /= norm