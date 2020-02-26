import numpy as np
import cv2
from matplotlib import pyplot as plt

# Convolv function for convolving the kernel with real image matrix
def convolve_np(image, kernel):
    X_height = image.shape[0]
    X_width = image.shape[1]

    F_height = kernel.shape[0]
    F_width = kernel.shape[1]
    
    H = int((F_height - 1) / 2)
    W = int((F_width - 1) / 2)
    
    out = np.zeros((X_height, X_width))
    
    for i in np.arange(H, X_height-H):
        for j in np.arange(W, X_width-W):
            sum = 0
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    a = image[i+k, j+l]
                    w = kernel[H+k, W+l]
                    sum += (w * a)
            out[i,j] = sum
        
    return out # Returning the Convolved Image

def box_blur():
    img = cv2.imread('Lenna.png', 0)
    
    # Sobel Operator for Horizontal Edge Detection
    Hx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    # Sobel Operator for Vertical Edge Detection
    Hy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    img_x = convolve_np(img, Hx) / 8.0
    img_y = convolve_np(img, Hy) / 8.0
    
    img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

    img_out = (img_out / np.max(img_out)) * 255 # Output of Canny Edge Detection
    
    plt.imshow(img_out, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    
if __name__ == '__main__':
    box_blur()
