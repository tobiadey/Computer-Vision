'''
Sharpening
A high-pass filter can be used to make an image appear sharper.
These filters emphasize fine details in the image - the opposite of the low-pass filter.
High-pass filtering works in the same way as low-pass filtering; it just uses a different convolution kernel.

As we have seen in the lecture,
sharpening can be performed by adding to the original image sharp details,
which can be computed as the difference between the original image and its blurred version.
These details are then scaled, and added back to the original image:
enhanced image = original + amount * (original - blurred)


This technique is also referred to as unsharp masking.
The blurring step could use any image filter method, e.g. average filter, but traditionally a gaussian filter is used.
skimage has a dedicated function called filters.unsharp_mask.
The radius parameter in the unsharp masking filter refers to the sigma parameter of the gaussian filter.
'''
from skimage import io, color, filters
import matplotlib.pyplot as plt

img = io.imread('Westminster.jpg')
img = color.rgb2gray(img)

img_sharpened = filters.unsharp_mask(img, radius=1, amount=2)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original'), ax[0].set_axis_off()
ax[1].imshow(img_sharpened, cmap='gray')
ax[1].set_title('Sharpening filter'), ax[1].set_axis_off()
fig.tight_layout()
plt.show()
