'''
Smoothing
Low-pass filtering (aka smoothing or blurring) is employed to remove high spatial frequency noise from a digital image.
The low-pass filters usually employ moving window operator that affects one pixel of the image at a time,
changing its value by some function of a local region (window) of pixels.
The operator moves over the image to affect all the pixels in the image.
'''


# '''
# Moving average Way 1
# '''
# print("\n")
# print("Moving average way 1")
#
#
# # Let's start with a moving average (or box filter).
# # We will first implement it using the convolve function, for which the actual kernel needs to be implemented:
#
from skimage import io, color, img_as_float, img_as_ubyte
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import numpy as np

#
#
# k_size = 5
# kernel = np.ones((k_size, k_size), dtype=float)/(k_size**2)
# print("Kernel = ", "\n", kernel)
#
# img = io.imread('LondonEye.jpg')
# img = color.rgb2gray(img)
# print('Img data type =', img.dtype, ', with max channel value =', np.max(img))
#
# img_filtered = convolve(img, kernel, mode='nearest')
#
# fig, ax = plt.subplots(1, 2, figsize=(7, 7))
# ax[0].imshow(img, cmap='gray')
# ax[0].set_title('Original'), ax[0].set_axis_off()
# ax[1].imshow(img_filtered, cmap='gray')
# ax[1].set_title('Average filter'), ax[1].set_axis_off()
# fig.tight_layout()
# plt.show();
# plt.close();
# # If you look carefully, you will see that the image to the right is smoothed, with many details blurred in the process.
# # Try to run again the filtering above using different kernel sizes.
#
#
#
# '''
# Moving average Way 2
# '''
# print("\n")
# print("Moving average way 2")
#
# # a moving average can be implemented also using a skimage function called filters.rank.mean
#
from skimage.morphology import square, disk
from skimage import filters

se_square = square(5)
# print("Square (5) = ", "\n", se_square)
# img_filtered2 = img_as_float(filters.rank.mean(img_as_ubyte(img), selem=se_square))    # rank.mean does not accept float
#
# fig, ax = plt.subplots(1, 2, figsize=(7, 7))
# ax[0].imshow(img_filtered, cmap='gray')
# ax[0].set_title('Filtered with explicit convolution'), ax[0].set_axis_off()
# ax[1].imshow(img_filtered2, cmap='gray')
# ax[1].set_title('Filtered with average filter'), ax[1].set_axis_off()
# fig.tight_layout()
# plt.show();
# plt.close();
# # Same result (minus very small differences at the boundaries and due to dtype approximations)
#
#
# '''
# As you can see, the two outputs are essentially the same
# (to be precise, there are very small differences at the boundaries due to the different
#  way with which these two algorithms deal with them, but these are largely neglible).
#
# Let's try now a different structuring element:
# Way 3
# '''
# print("\n")
# print("Moving average way 3")
#
# se_disk_3 = disk(3)
# se_disk_5 = disk(5)
# print("Disk (3) = ", "\n", se_disk_3)
# print("Disk (5) = ", "\n", se_disk_5)
#
# img_filt_disk_3 = img_as_float(filters.rank.mean(img_as_ubyte(img), selem=se_disk_3))
# img_filt_disk_5 = img_as_float(filters.rank.mean(img_as_ubyte(img), selem=se_disk_5))
#
# fig, ax = plt.subplots(1, 3, figsize=(10, 10))
# ax[0].imshow(img, cmap='gray')
# ax[0].set_title('Original'), ax[0].set_axis_off()
# ax[1].imshow(img_filt_disk_3, cmap='gray')
# ax[1].set_title('Average filter (disk = 3)'), ax[1].set_axis_off()
# ax[2].imshow(img_filt_disk_5, cmap='gray')
# ax[2].set_title('Average filter (disk = 5)'), ax[2].set_axis_off()
# fig.tight_layout()
# plt.show();
# plt.close();
#

'''
low pass filters can be used for denoising. Let's test this.
First, to make the input a little bit dirty, we spray some salt & pepper noise on the image and check on the result:
'''
print("\n")
print("adding salt and pepper & filtering it out using Gaussian fitler")
from skimage import util, data

# img = img_as_float(data.camera())
#
# # Add salt & pepper noise
# img_noisy = util.random_noise(img, mode='s&p')
#
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax[0].imshow(img, cmap='gray')
# ax[0].set_title('Original'), ax[0].set_axis_off()
# ax[1].imshow(img_noisy, cmap='gray')
# ax[1].set_title('Noisy'), ax[1].set_axis_off()
# fig.tight_layout()
# plt.show();
#
# # Let's try to filter it using an average filter. We will also try a Gaussian filter, using the dedicated function:
#
# img_noisy_avg_filt = img_as_float(filters.rank.mean(img_as_ubyte(img_noisy), selem=se_square))
# img_noisy_gauss_filt = filters.gaussian(img_noisy, sigma=1)
#
# fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# ax[0].imshow(img_noisy, cmap='gray')
# ax[0].set_title('Noisy'), ax[0].set_axis_off()
# ax[1].imshow(img_noisy_avg_filt, cmap='gray')
# ax[1].set_title('Average filter'), ax[1].set_axis_off()
# ax[2].imshow(img_noisy_gauss_filt, cmap='gray')
# ax[2].set_title('Gaussian filter sigma =1'), ax[2].set_axis_off()
# fig.tight_layout()
# plt.show();

'''
None of the two filters did actually remove the noise: they just blurred it.
Task 1.1 Filtering out salt & pepper noise
'''

##############################################################################
# TODO: Filter salt & pepper noise and show results
##############################################################################

# img_noisy_medi_filt = img_as_float(filters.rank.median(img_as_ubyte(img_noisy), selem=se_square))
# img_noisy_gauss2_filt = filters.gaussian(img_noisy, sigma=5)
#
#
#
# fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# ax[0].imshow(img_noisy, cmap='gray')
# ax[0].set_title('Noisy'), ax[0].set_axis_off()
# ax[1].imshow(img_noisy_medi_filt, cmap='gray')
# ax[1].set_title('Median filter'), ax[1].set_axis_off()
# ax[2].imshow(img_noisy_gauss2_filt, cmap='gray')
# ax[2].set_title('Gaussian filter sigma = 5'), ax[2].set_axis_off()
# fig.tight_layout()
# plt.show();

# avg - filters.rank.mean doesnt seem like theres anything i can do to change it, maybe there are some extra params.
# Gaussian - setting sigma = 5 reducesthe salt & pepper but the image is very blurry.

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################


'''
Taks 1.2 Filtering colour images
'''

'''
way 1 failed
'''
# img = io.imread('LondonEye.jpg')
# print('Img1 data type =', img.dtype, ', with max channel value =', np.max(img))
#
# ##############################################################################
# # TODO: Add noise to the RGB image, then implement median filter
# ##############################################################################
# # Add salt & pepper noise
# img_noisy2 = util.random_noise(img, mode='s&p')
# print('Img2 data type =', img_noisy2.dtype, ', with max channel value =', np.max(img_noisy2))
#
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax[0].imshow(img, cmap='gray')
# ax[0].set_title('Original'), ax[0].set_axis_off()
# ax[1].imshow(img_noisy2, cmap='gray')
# ax[1].set_title('Noisy'), ax[1].set_axis_off()
# fig.tight_layout()
# plt.show();
#
# # median filter init
# img_filtered_medi = img_as_float(filters.rank.median(img_as_ubyte(img_noisy),disk(5))
# print('Img3 data type =', img_filtered_medi.dtype, ', with max channel value =', np.max(img_filtered_medi))
#
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax[0].imshow(img_noisy2)
# ax[0].set_title('Noisy'), ax[0].set_axis_off()
# ax[1].imshow(img_filtered_medi)
# ax[1].set_title('Median filter'), ax[1].set_axis_off()
# fig.tight_layout()
# plt.show();

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
'''
way 2 solution
'''
img = io.imread('LondonEye.jpg')

##############################################################################
# TODO: Add noise to the RGB image, then implement median filter
##############################################################################
img_noisy2 = util.random_noise(img, mode='s&p')

print('Img2 data type =', img_noisy2.dtype, ', with max channel value =', np.max(img_noisy2))

r_ch = img_noisy2[:, :, 0]
g_ch = img_noisy2[:, :, 1]
b_ch = img_noisy2[:, :, 2]

# median filter init
r_ch_median = filters.median(r_ch, selem=square(3))
g_ch_median = filters.median(g_ch, selem=square(3))
b_ch_median = filters.median(b_ch, selem=square(3))

img_median_filt = np.dstack((r_ch_median, g_ch_median, b_ch_median))
# img_filtered_medi = img_as_float(filters.median(img_as_ubyte(img_noisy2)),disk(5))
# print('Img3 data type =', img_filtered_medi.dtype, ', with max channel value =', np.max(img_filtered_medi))

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(img_noisy2)
ax[0].set_title('Noisy'), ax[0].set_axis_off()
ax[1].imshow(img_median_filt)
ax[1].set_title('Median filter'), ax[1].set_axis_off()
fig.tight_layout()
plt.show();

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################


'''
Task 1.3 (Optional): Filtering specific areas of an image

I will use gaussian filteting and set a high sigma
'''
img = io.imread('Class.jpg')

##############################################################################
# TODO: Blur the faces in the image and show results
##############################################################################
img_blurred = img_as_ubyte(filters.gaussian(img, sigma=10, multichannel=True))
face_coords = np.array([[270, 388, 107, 240],
                        [180, 295, 480, 575],
                        [150, 242, 809, 893]])
img_composite = img.copy()

for i in range(face_coords.shape[0]):
    img_composite[face_coords[i, 0]:face_coords[i, 1], face_coords[i, 2]:face_coords[i, 3], :] = \
        img_blurred[face_coords[i, 0]:face_coords[i, 1], face_coords[i, 2]:face_coords[i, 3], :]

plt.figure(figsize=(12, 9))
plt.imshow(img_composite)
plt.show()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
