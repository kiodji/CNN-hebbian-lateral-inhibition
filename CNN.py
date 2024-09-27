from numba import jit, cuda 
import numpy as np 
from OnOffCells import OnOffCells
from Layer import Layer
from FCLayer import FullyConnectedLayer
from MaxPooling import MaxPooling
import random
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
class CNN:
    def __init__(self):
        self.images = []
        self.imagesTest = []
        self.labels = [None] * 20
        self.labelsTest = [None] * 16
        self.image = None
        self.imgCount = 0
        self.testImgCount = 0
        self.output = None
        self.lastLayer = None
        self.fc_input_size = None
        self.total_predictions = 0
        self.correct_predictions = 0
        
        self.labelNames = {
            0: 'Human',
            1: 'Cat',
            2: 'Mouse',
            3: 'Dog'
        }

        # Layers  
        self.conv1 = Layer(32, 1, 3, learning_rate=1.5, stride=1)
        self.conv2 = Layer(32, 32, 5, learning_rate=1.5, stride=1)
        self.conv3 = Layer(32, 32, 7, learning_rate=1.5)

    def addImage(self, path, label):
        image = Image.open(path).convert('L')
        self.images.append(image)
        self.labels[self.imgCount] = label
        self.imgCount += 1

    def addTestImage(self, path, label):
        image = Image.open(path).convert('L')
        self.imagesTest.append(image)
        self.labelsTest[self.testImgCount] = label
        self.testImgCount += 1

    def run(self):

        r = random.randint(0, len(self.images) - 1)
        self.image = np.array(self.images[r]).reshape(1, self.images[r].height, self.images[r].width) / 255
        # self.image = np.array(self.images[r])

        self.output = self.conv1.forward(self.image)

        self.output = self.conv2.forward(self.output)

        self.output = self.conv3.forward(self.output)

        if self.fc_input_size is None:
            self.fc_input_size = self.conv3.filters_num * self.conv3.last_output.shape[1] * self.conv3.last_output.shape[2]
            self.fc = FullyConnectedLayer(self.fc_input_size, 4, learning_rate=1)


        fc_input = self.output.flatten()
        
        probabilities = self.fc.forward(fc_input)
        
        # Create target array (one-hot encoding)
        target = np.zeros(probabilities.shape)
        target[self.labels[r]] = 1 
        
        self.fc.backward(target)
        # print("current label: " + str(r))
        # print(probabilities)

        predicted_label = np.argmax(probabilities)
        true_label = self.labels[r]

        self.total_predictions += 1
        if predicted_label == true_label:
            self.correct_predictions += 1

        print(f"current label: {self.labelNames[true_label]}, predicted label: {self.labelNames[predicted_label]}")
        print(probabilities)
        print(f"Accuracy: {(self.correct_predictions / self.total_predictions) * 100:.2f}%")

        self.lastLayer = self.conv3

        # print(self.conv1.filters)

    def test(self):

        r = random.randint(0, len(self.imagesTest) - 1)
        self.image = np.array(self.imagesTest[r]).reshape(1, self.imagesTest[r].height, self.imagesTest[r].width) / 255
        # self.image = np.array(self.imagesTest[r])
        
        self.output = self.conv1.forward(self.image)

        self.output = self.conv2.forward(self.output)

        self.output = self.conv3.forward(self.output)

        if self.fc_input_size is None:
            self.fc_input_size = self.conv3.filters_num * self.conv3.last_output.shape[1] * self.conv3.last_output.shape[2]
            self.fc = FullyConnectedLayer(self.fc_input_size, 5, learning_rate=0.01)

        fc_input = self.output.flatten()
        
        probabilities = self.fc.forward(fc_input)
        
        predicted_label = np.argmax(probabilities)
        true_label = self.labelsTest[r]

        self.total_predictions += 1
        if predicted_label == true_label:
            self.correct_predictions += 1

        print(f"current label: {self.labelNames[true_label]}, predicted label: {self.labelNames[predicted_label]}")
        print(probabilities)
        print(f"Accuracy: {(self.correct_predictions / self.total_predictions) * 100:.2f}%")

        self.lastLayer = self.conv3
    
    def get_kernels_images(self):
        kernel_image_group = []

        for f in range(self.lastLayer.filters_num):
        # for k in range(self.lastLayer.depth):

            kernel = self.lastLayer.filters[f, 0, :, :]
            kernel = (kernel * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(kernel, mode='L')

            enlarged_image = image.resize((180, 180), Image.NEAREST)

            kernel_image_group.append(enlarged_image)

        return kernel_image_group
    
    def get_output_images(self):

        new_height = 180
        new_width = 180

        image_group = []

        for f in range(self.lastLayer.filters_num):
            pixels = self.lastLayer.last_output[f, :, :]

            pixels = pixels.clip(0, 255).astype(np.uint8) * 255
            image = Image.fromarray(pixels)

            enlarged_image = image.resize((new_width, new_height), Image.NEAREST)

            image_group.append(enlarged_image)

        return image_group
        
