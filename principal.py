#importing required libraries
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure, color
import matplotlib.pyplot as plt
import numpy as np

# reading the image
img = imread('cat.jpg')
plt.axis("off")
plt.imshow(img)
plt.show()
print(img.shape)

resized_img = resize(img, (128 * 4, 64 * 4))
plt.axis("off")
plt.imshow(resized_img)
plt.show()
print(resized_img.shape)

# Convert to grayscale if the image is colored
gray_img = color.rgb2gray(resized_img)

# creating hog features
fd, hog_image = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)

# Normalizing images to uint8
resized_img_uint8 = (resized_img * 255).astype(np.uint8)
hog_image_uint8 = (hog_image * 255).astype(np.uint8)

# Displaying the HOG image
plt.axis("off")
plt.imshow(hog_image_uint8, cmap="gray")
plt.show()

# saving images
imsave("resized_img.jpg", resized_img_uint8)
imsave("hog_image.jpg", hog_image_uint8, cmap="gray")
