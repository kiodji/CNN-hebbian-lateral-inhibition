from numba import cuda
import numpy as np

class OnOffCells:
    def __init__(self):
        self.stride = 1
        self.receptive_field_size = 3
        self.depth = 1
        self.filters_num = 2
        self.weights = np.random.uniform(0, 1, (self.filters_num, self.depth, self.receptive_field_size, self.receptive_field_size)).astype(np.float32)
        self.last_input = None
        self.last_output = None
        self.max_filter = None

    def forward(self, input):
        self.last_input = input
        height, width = input.shape

        filter_maps = np.zeros((self.filters_num, height, width), dtype=np.float32)

        # Transfer input to GPU
        d_input = cuda.to_device(self.last_input)
        d_filter_maps = cuda.to_device(filter_maps)

        # Define the kernel launch dimensions
        threads_per_block = (16, 16)
        blocks_per_grid_x = int(np.ceil(height / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(width / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch the synapticIntegration kernel
        synapticIntegration_kernel[blocks_per_grid, threads_per_block](d_input, d_filter_maps, self.filters_num, self.receptive_field_size, height, width, self.stride)
        
        # Copy the results back to the host
        d_filter_maps.copy_to_host(filter_maps)

        self.last_output = filter_maps
        return self.last_output

@cuda.jit
def synapticIntegration_kernel(input, filter_maps, filters_num, receptiveFieldSize, height, width, stride):
    i, j = cuda.grid(2)
    paddingSize = receptiveFieldSize // 2

    if i >= paddingSize and i < height - paddingSize and j >= paddingSize and j < width - paddingSize:
        central_pixel = input[i + paddingSize, j + paddingSize]  # Central pixel
        surround_sum = 0.0

        # Calculating the contrast difference between the central pixel and surrounding pixels
        for a in range(i - paddingSize, i + paddingSize):
            for b in range(j - paddingSize, j + paddingSize):
                if a >= width or a >= height or b >= width or b >= height:
                    continue
                if a != i and b != j:  # Exclude the central pixel
                    surround_sum += input[a, b]

        # Calculate ON and OFF signals for RGC ON-OFF cells
        on_center_sum = max(central_pixel - surround_sum, 0)
        off_center_sum = max(surround_sum - central_pixel, 0)

        filter_maps[0, i, j] = input[i, j]
        # filter_maps[0, i, j] = on_center_sum
        # filter_maps[1, i, j] = off_center_sum