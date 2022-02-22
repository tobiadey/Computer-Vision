'''
Lecture 2
load and display images
explore some simple image processing methods, including some Instagram-like filters and contrast adjustments.
'''



'''
1. Negative four ways
create the negative of a given image. We will do it in four different ways, so to better understand how pixels are processed.
'''

# # load the image using the skimage package:
# from skimage.io import imread
# img = imread('Peppers.jpg')
# # "Surinamese peppers" by Daveness_98 is licensed under CC BY 2.0
#
# # plot image
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show();
#

#
# # WAY1
# # the fastest way to produce the negative image is the following:
# # Here, the value of each R, G, and B component at every pixel in img is subtracted from 255.
# img_neg = 255 - img
# plt.imshow(img_neg)
# plt.show();
# print("negative image way 1")
#
# # WAY2
# # another method now to achieve the same result.
# # An image is a collection of pixels (for a colour image, each having a red, green, and blue component).
# # The pixels are in the form of a 2D grid.
# # If you wish, you can loop over all these pixels, using a doubly-nested loop (loop within a loop)
# # and invert each channel separately.
#
# # First we can extract the size of the grid:
# [height, width, channels] = img.shape
# print('Image dimensions =', img.shape)
#
# img_neg = img.copy()        # Create a copy of img
#
# # And now we can loop along both dimensions:
# for y in range(height):
#   for x in range(width):
#     img_neg[y, x, 0] = 255 - img[y, x, 0]      # R
#     img_neg[y, x, 1] = 255 - img[y, x, 1]      # G
#     img_neg[y, x, 2] = 255 - img[y, x, 2]      # B
#
# plt.imshow(img_neg)
# plt.show();
# print("negative image way 2")
#
#
# # WAY3
# # Let's try a third method.
# # We’ll still use a doubly-nested loop to loop over all the pixels in the image.
# # However, since inverting an image applies the same operation to the red, green, and blue channels,
# # we can use the colon operator to access the third dimension in the image (the colour at this pixel)
# # and process the three colour channels with a single line of code:
#
# img_neg = img.copy()
#
# for y in range(height):
#   for x in range(width):
#     img_neg[y, x, :] = 255 - img[y, x, :]      # All three channels at once
#
# plt.imshow(img_neg)
# plt.show();
# print("negative image way 3")
#
#
# # WAY4
# from skimage.util import invert
#
# img_neg = invert(img)
# plt.imshow(img_neg)
# plt.show();
# print("negative image way 4")
#
#
#




'''
1.1 Swapping colours
'''
#
# # TASK:
# # access each pixel's colour channel, try to swap colours.
# # Specifically, swap the blue and red channels, while keeping the green unchanged.
#
# img_swap = img.copy()
#
# # TODO: Swap blue and red channels, keeping green unchanged
# [height, width, channels] = img.shape
# print('Image dimensions =', img.shape)
#
# for y in range(height):
#   for x in range(width):
#     img_swap[y, x, 0] = img[y, x, 2]     # R
#     img_swap[y, x, 2] = img[y, x, 0]     # B
#
# plt.imshow(img_swap)
# plt.show()


'''
2 Instagram filters
'''
# # Let's start by loading and displaying an image with a nice selfie:
# from skimage.io import imread
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# img_rgb = imread('DogSelfie.jpg')
# # #dog #selfie" by nchenga is licensed under CC BY-NC 2.0
#
# plt.figure(figsize=(7, 7))     # Set figure size in inches
# plt.imshow(img_rgb)
# plt.show();
#
# # we will use an image processing method called tinting,
# # this usually consists in applying a multiplying coefficient to one or more colour channels.
#
# # we can perform tinting in the RGB space.
# # However, it can be easily applied to other colour spaces too.
# # For instances, what if we wanted to double the saturation of the image?
#
# # The first stpe would consist in performing the colour space conversion from RGB to HSV.
# # skimage provides a handy function for that,
# # but watch out: these conversions often trigger a data type change to avoid losses of precision.
#
# from skimage import color, img_as_float, img_as_ubyte
#
# img_hsv = color.rgb2hsv(img_rgb)
# print('Img_rgb data type =', img_rgb.dtype, ', with max channel value =', np.max(img_rgb))
# print('Img_hsv data type =', img_hsv.dtype, ', with max channel value =', np.max(img_hsv))
#
# # The next step consists in the actual tinting operation, just as we did in the RGB space in the lecture:
# s_channel = img_hsv[:, :, 1]
# s_channel = s_channel * 2    # Rescale S channel
# s_channel[s_channel>1] = 1   # Clip values > 1
# img_hsv[:, :, 1] = s_channel
#
# # Now we can go back to the RGB space (and also to uint8 data type):
# img_rgb_sat = color.hsv2rgb(img_hsv)
# img_rgb_sat = img_as_ubyte(img_rgb_sat)
# print('Img_rgb_sat data type =', img_rgb_sat.dtype, ', with max channel value =', np.max(img_rgb_sat))
#
# fig, ax = plt.subplots(1, 2, figsize=(7, 7))
# ax[0].imshow(img_rgb)
# ax[1].imshow(img_rgb_sat)
# plt.show();
#



'''
2.1 Adding a vignette
'''
# Another common effect adopted in Instagram filters is vignetting,
# which consists in lowering the intensity of the image as we go from the centre towards the corners.
# This helps draw the attention to the centre of the image, and gives the image a certain low fidelity look.

# Let's first load another nice image (specifically,a close-up of Chelsea the cat, from the example data available in skimage)
# and already create a copy image, which will be useful when implementing the vignette

from skimage import data
import matplotlib.pyplot as plt
from math import e
import numpy
from numpy import linalg as LA

cat = data.chelsea()
cat_vign = cat.copy()

# plt.imshow(cat)
# plt.show()


##############################################################################
# TODO: Add a vignette1
##############################################################################
[height, width, channels] = cat.shape
print('Image dimensions =', cat.shape)

# identify the centre of the image.
midX,midY = width/2,height/2
print(midX,midY)
print("this is e:", e)

# print(e**0)
# print(e**-1)
# r = 0   #distance of a given pixel from the centre

# compute a function that will modify the brightness from the centre of the image
def modify_brightness(r,x,y):
    # f = e ** (-(r/width))
    f = numpy.exp((-(r/width)))

    # use f to scale the image brightness
    # by multiplying the image’s R, G, and B components or V component if in HSV

    # if(pos == pos):
    cat_vign[y, x, :] = f* cat[y, x, :]      # R
    # cat_vign[y, x, 1] = f* cat[y, x, 1]      # G
    # cat_vign[y, x, 2] = f* cat[y, x, 2]      # B



a = numpy.array((midY ,midX))
# form a doubly-nested loop to loop over every pixel in the image
# In the inner part of the loop, compute the radius r,
for y in range(height):
  for x in range(width):
    b = numpy.array((y,x))
    r = LA.norm(a-b)
    # non negative why?
    print(r)
    modify_brightness(r,x,y)

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################


fig, ax = plt.subplots(1, 2, figsize=(7, 7))
ax[0].imshow(cat)
ax[1].imshow(cat_vign)
plt.show()


##############################################################################
# TODO: Add a vignette2
##############################################################################
[height, width, _] = cat.shape
c_coord = numpy.array([height, width])/2

for y in range(height):
  for x in range(width):
    px_coord = numpy.array([y, x])
    r = numpy.linalg.norm(px_coord-c_coord)
    f = numpy.exp(-r/width)
    cat_vign[y, x, :] = cat[y, x, :] * f

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

fig, ax = plt.subplots(1, 2, figsize=(7, 7))
ax[0].imshow(cat)
ax[1].imshow(cat_vign)
plt.show()



'''
3 Contrast adjustments
'''

# define a function that will display both image and histogram. Make sure to understand what each line does!
from skimage import img_as_float

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram.
    """
    image = img_as_float(image)
    ax_img, ax_hist = axes

    # Display image
    ax_img.imshow(image, cmap='gray')
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    return ax_img, ax_hist
#
# # Now let's load a grayscale image from the example data available in skimage
# from skimage import data, exposure
# import matplotlib.pyplot as plt
# import numpy as np
#
# img = img_as_float(data.moon())
#
# plt.imshow(img, cmap='gray')
# plt.show()
#
# # Let's now implement three different contrast adjustments methods:
# # contrast stretching,
# # gamma correction and
# # histogram equalisation.
#
# # For the first one,
# # we will stretch the histogram so that the 2nd and 98th percentile will respectively become the new 0 and 1 values.
# # For the second, we will use a gamma value of 2.
#
# # Contrast stretching
# p2, p98 = np.percentile(img, (2, 98))
# img_contrast_stretch = exposure.rescale_intensity(img, in_range=(p2, p98))
#
# # Gamma correction
# gamma = 2
# img_gamma = exposure.adjust_gamma(img, gamma)
#
# # Histogram equalisation
# img_hist_eq = exposure.equalize_hist(img)
#
#
# # Let's now plot the results:
# fig = plt.figure(figsize=(10, 6))
# axes = np.zeros((2, 4), dtype=np.object)
# axes[0, 0] = plt.subplot(2, 4, 1)
# axes[0, 1] = plt.subplot(2, 4, 2, sharex=axes[0, 0], sharey=axes[0, 0])
# axes[0, 2] = plt.subplot(2, 4, 3, sharex=axes[0, 0], sharey=axes[0, 0])
# axes[0, 3] = plt.subplot(2, 4, 4, sharex=axes[0, 0], sharey=axes[0, 0])
# axes[1, 0] = plt.subplot(2, 4, 5)
# axes[1, 1] = plt.subplot(2, 4, 6)
# axes[1, 2] = plt.subplot(2, 4, 7)
# axes[1, 3] = plt.subplot(2, 4, 8)
#
# ax_img, ax_hist = plot_img_and_hist(img, axes[:, 0])
# ax_img.set_title('Original image')
#
# y_min, y_max = ax_hist.get_ylim()
# ax_hist.set_ylabel('Number of pixels')
# ax_hist.set_yticks(np.linspace(0, y_max, 5))
#
# ax_img, ax_hist = plot_img_and_hist(img_contrast_stretch, axes[:, 1])
# ax_img.set_title('Contrast stretching')
#
# ax_img, ax_hist = plot_img_and_hist(img_gamma, axes[:, 2])
# ax_img.set_title('Gamma correction')
#
# ax_img, ax_hist = plot_img_and_hist(img_hist_eq, axes[:, 3])
# ax_img.set_title('Histogram equalisation')
#
# fig.tight_layout()     # prevent overlap of y-axis labels
# plt.show()

'''
3.1 Changing contrast to a different image
'''
# # Substitute the moon image with Chelsea the cat.
# # Convert it to grayscale, and test the previous brightness transformations on the image.
# ##############################################################################
# # TODO: Test the previous transformations on a different image
# ##############################################################################
# from skimage import data, exposure
# from skimage.color import rgb2gray
# import matplotlib.pyplot as plt
# import numpy as np
#
# img = img_as_float(data.chelsea())
# img = rgb2gray(img)
#
# plt.imshow(img, cmap='gray')
# plt.show()
#
# # Contrast stretching
# p2, p98 = np.percentile(img, (2, 98))
# img_contrast_stretch = exposure.rescale_intensity(img, in_range=(p2, p98))
#
# # Gamma correction
# gamma = 2
# img_gamma = exposure.adjust_gamma(img, gamma)
#
# # Histogram equalisation
# img_hist_eq = exposure.equalize_hist(img)
#
# fig = plt.figure(figsize=(10, 6))
# axes = np.zeros((2, 4), dtype=np.object)
# axes[0, 0] = plt.subplot(2, 4, 1)
# axes[0, 1] = plt.subplot(2, 4, 2, sharex=axes[0, 0], sharey=axes[0, 0])
# axes[0, 2] = plt.subplot(2, 4, 3, sharex=axes[0, 0], sharey=axes[0, 0])
# axes[0, 3] = plt.subplot(2, 4, 4, sharex=axes[0, 0], sharey=axes[0, 0])
# axes[1, 0] = plt.subplot(2, 4, 5)
# axes[1, 1] = plt.subplot(2, 4, 6)
# axes[1, 2] = plt.subplot(2, 4, 7)
# axes[1, 3] = plt.subplot(2, 4, 8)
#
# ax_img, ax_hist = plot_img_and_hist(img, axes[:, 0])
# ax_img.set_title('Original image')
#
# y_min, y_max = ax_hist.get_ylim()
# ax_hist.set_ylabel('Number of pixels')
# ax_hist.set_yticks(np.linspace(0, y_max, 5))
#
# ax_img, ax_hist = plot_img_and_hist(img_contrast_stretch, axes[:, 1])
# ax_img.set_title('Contrast stretching')
#
# ax_img, ax_hist = plot_img_and_hist(img_gamma, axes[:, 2])
# ax_img.set_title('Gamma correction')
#
# ax_img, ax_hist = plot_img_and_hist(img_hist_eq, axes[:, 3])
# ax_img.set_title('Histogram equalisation')
#
# fig.tight_layout()     # prevent overlap of y-axis labels
# plt.show()
# ##############################################################################
# #                             END OF YOUR CODE                               #
# ##############################################################################
#



print("THE END")
