'''
Edge detection.
Edge detection is a fundamental step of many computer vision pipelines.
There are several methods for edge detection in images:
in this section we will perform detection of vertical and horizontal edges in an image using first order derivatives,
and also use Sobel kernels. We will finally compute the image Laplacian (which uses second order derivatives).

'''

# Let's start by creating the first derivative kernel along the x direction exactly as we did in the lecture:
from skimage import io, color, filters
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import numpy as np

img = io.imread('Westminster.jpg')
img = color.rgb2gray(img)

kernel_x = 0.5*np.array([-1, 0, 1]).reshape((1, 3))
print("Kernel = ", "\n", kernel_x)

img_x = convolve(img, kernel_x)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original'), ax[0].set_axis_off()
ax[1].imshow(img_x, cmap='gray')
ax[1].set_title('X Derivative'), ax[1].set_axis_off()
fig.tight_layout()
plt.show();

kernel_y = kernel_x.T
print("Kernel = ", "\n", kernel_y)

img_y = convolve(img, kernel_y)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original'), ax[0].set_axis_off()
ax[1].imshow(img_y, cmap='gray')
ax[1].set_title('Y Derivative'), ax[1].set_axis_off()
fig.tight_layout()
plt.show();

# Notice the difference between the two image derivatives.
# In a similar fashion, we could implement other types of first derivative kernels.

'''
However,
skimage has a series of filtering functions already implemented for the most common variants
(e.g. Sobel, Prewitt):
'''

img_sobel_x = filters.sobel_v(img)
img_sobel_y = filters.sobel_h(img)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(img_sobel_x, cmap='gray')
ax[0].set_title('X Derivative (Sobel)'), ax[0].set_axis_off()
ax[1].imshow(img_sobel_y, cmap='gray')
ax[1].set_title('Y Derivative (Sobel)'), ax[1].set_axis_off()
fig.tight_layout()
plt.show();


'''
Task 3.1: Comparing derivative kernels

How do the different kernel designs compare?
Compute the gradient magnitude of the image using the standard image derivatives,
and compare the result to the gradient magnitude computed using the Sobel derivatives.
For the latter, note that there's no need for you to use the two component separately:
you can use the sobel function in skimage.
'''

##############################################################################
# TODO: Compute gradient magnitude and compare to Sobel one
##############################################################################


img_gradient_mag = np.sqrt(img_x**2 + img_y**2)
img_sobel_gradient_mag = filters.sobel(img)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(img_gradient_mag, cmap='gray')
ax[0].set_title('gradient magnitude'), ax[0].set_axis_off()
ax[1].imshow(img_sobel_gradient_mag, cmap='gray')
ax[1].set_title('gradient magnitude (Sobel)'), ax[1].set_axis_off()
fig.tight_layout()
plt.show();

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################


img_laplacian = filters.laplace(img)

fig, ax = plt.subplots(1, 2, figsize=(14, 8))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original'), ax[0].set_axis_off()
ax[1].imshow(img_laplacian, cmap='gray')
ax[1].set_title('Image Laplacian'), ax[1].set_axis_off()
fig.tight_layout()
plt.show();
