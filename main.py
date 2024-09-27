import numpy as np
from CNN import CNN
from timeit import default_timer as timer    
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

cnn = CNN()

cnn.addImage("Images\\human\\1.jpg", 0)
cnn.addImage("Images\\human\\2.jpg", 0)
cnn.addImage("Images\\human\\3.jpg", 0)
cnn.addImage("Images\\human\\4.jpg", 0)
cnn.addImage("Images\\human\\5.jpg", 0)

cnn.addImage("Images\\cat\\1.jpg", 1)
cnn.addImage("Images\\cat\\2.jpg", 1)
cnn.addImage("Images\\cat\\3.jpg", 1)
cnn.addImage("Images\\cat\\4.jpg", 1)
cnn.addImage("Images\\cat\\5.jpg", 1)

cnn.addImage("Images\\mouse\\1.jpg", 2)
cnn.addImage("Images\\mouse\\2.jpg", 2)
cnn.addImage("Images\\mouse\\3.jpg", 2)
cnn.addImage("Images\\mouse\\4.jpg", 2)
cnn.addImage("Images\\mouse\\5.jpg", 2)

cnn.addImage("Images\\dog\\1.jpg", 3)
cnn.addImage("Images\\dog\\2.jpg", 3)
cnn.addImage("Images\\dog\\3.jpg", 3)
cnn.addImage("Images\\dog\\4.jpg", 3)
cnn.addImage("Images\\dog\\5.jpg", 3)

# test images

cnn.addTestImage("Images\\test\\human\\1.jpg", 0)
cnn.addTestImage("Images\\test\\human\\2.jpg", 0)
cnn.addTestImage("Images\\test\\human\\3.jpg", 0)
cnn.addTestImage("Images\\test\\human\\4.jpg", 0)

cnn.addTestImage("Images\\test\\cat\\1.jpg", 1)
cnn.addTestImage("Images\\test\\cat\\2.jpg", 1)
cnn.addTestImage("Images\\test\\cat\\3.jpg", 1)
cnn.addTestImage("Images\\test\\cat\\4.jpg", 1)

cnn.addTestImage("Images\\test\\mouse\\1.jpg", 2)
cnn.addTestImage("Images\\test\\mouse\\2.jpg", 2)
cnn.addTestImage("Images\\test\\mouse\\3.jpg", 2)
# cnn.addTestImage("Images\\test\\mouse\\4.jpg", 2)
# cnn.addTestImage("Images\\test\\mouse\\5.jpg", 2)

cnn.addTestImage("Images\\test\\dog\\1.jpg", 3)
cnn.addTestImage("Images\\test\\dog\\2.jpg", 3)
cnn.addTestImage("Images\\test\\dog\\3.jpg", 3)
cnn.addTestImage("Images\\test\\dog\\4.jpg", 3)
cnn.addTestImage("Images\\test\\dog\\5.jpg", 3)


# cnn.addImage("Images\\0.png", 0)
# cnn.addImage("Images\\1.png", 1)
# cnn.addImage("Images\\2.png", 2)
# cnn.addImage("Images\\3.png", 3)
# cnn.addImage("Images\\4.png", 4)

# cnn.addImage("Images\\0_1.png", 0)
# cnn.addImage("Images\\1_1.png", 1)
# cnn.addImage("Images\\2_1.png", 2)
# cnn.addImage("Images\\3_1.png", 3)
# cnn.addImage("Images\\4_1.png", 4)

# cnn.addImage("Images\\0_2.png", 0)
# cnn.addImage("Images\\1_2.png", 1)
# cnn.addImage("Images\\2_2.png", 2)
# cnn.addImage("Images\\3_2.png", 3)
# cnn.addImage("Images\\4_2.png", 4)

# cnn.addImage("Images\\0_3.png", 0)
# cnn.addImage("Images\\1_3.png", 1)
# cnn.addImage("Images\\2_3.png", 2)
# cnn.addImage("Images\\3_3.png", 3)
# cnn.addImage("Images\\4_3.png", 4)
def runButton():
    cnn.run()
    display_images()

def testButton():
    cnn.test()
    display_images()

root = tk.Tk()
root.title("Model")
root.geometry("800x600")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

button = tk.Button(root, text="Run", command=runButton)
button.pack(side=tk.TOP, anchor='ne', padx=10, pady=10)

button = tk.Button(root, text="Test", command=runButton)
button.pack(side=tk.TOP, anchor='ne', padx=10, pady=10)

scrollbar_y = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

scrollbar_x = tk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
scrollbar_x.pack(side=tk.TOP, fill=tk.X)

canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

second_frame = tk.Frame(canvas)

canvas.create_window((0,0), window=second_frame, anchor="nw")

def display_images():

    for widget in second_frame.winfo_children():
        widget.destroy()

    kernels = cnn.get_kernels_images()
    images = cnn.get_output_images()
    num_kernels = len(kernels)
    num_images = len(images)

    num_cols_kernels = (num_kernels + 1) // 2
    num_cols_images = (num_images + 1) // 2

    for i in range(num_kernels):
        img = ImageTk.PhotoImage(kernels[i])
        label = tk.Label(second_frame, image=img)
        label.image = img
        label.grid(row=i // num_cols_kernels, column=i % num_cols_kernels, padx=2, pady=2)

    start_row = 2
    for i in range(num_images):
        img = ImageTk.PhotoImage(images[i])
        label = tk.Label(second_frame, image=img)
        label.image = img
        label.grid(row=start_row + (i // num_cols_images), column=i % num_cols_images, padx=2, pady=2)


root.mainloop()
