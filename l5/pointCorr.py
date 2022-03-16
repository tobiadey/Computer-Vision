'''
Lecture 5 
Image matching
intro
'''




'''
Image matching with point correspondences
Image matching consists in finding the transformation (affine, projective, etc.) that aligns one image to another. 
this task can be achieved by manually selecting correponding points in the two images.

Let's see how this can be done using skimage. 
First, let's load an image from the ones provided with the library, 
and apply an affine transformation to it to create the warped version. 
Our goal will be to find the transformation that aligns the warped image back to the original one.
'''
from skimage import data, transform
import matplotlib.pyplot as plt
import numpy as np


img = data.camera()

# define custom affine transformation object
tform = transform.AffineTransform(scale=(0.9, 0.9), rotation=0.2, translation=(20, -10))

# warp image
img_warped = transform.warp(img, tform.inverse)

fig, ax = plt.subplots(1, 2, figsize=(9, 6))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original'), ax[0].axis('off')
ax[1].imshow(img_warped, cmap='gray')
ax[1].set_title('Warped'), ax[1].axis('off')
fig.tight_layout()
plt.show()

# At this point, we should use our pointer to identify the  (𝑥,𝑦 ) coordinates 
# of a series of (at least) 3 points in both images. 
# For your convenience, I have already done this:
img_points = np.array([[135, 365], [236, 150], [417, 129], [331, 304]])
img_warped_points = np.array([[73, 335], [205, 163], [365, 180], [260, 319]])

# Let's see where these points are:
fig, ax = plt.subplots(1, 2, figsize=(9, 6))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original'), ax[0].axis('off')
ax[0].plot(img_points[:, 0], img_points[:, 1], 'xr')
ax[1].imshow(img_warped, cmap='gray')
ax[1].plot(img_warped_points[:, 0], img_warped_points[:, 1], 'xb')
ax[1].set_title('Warped'), ax[1].axis('off')
fig.tight_layout()
plt.show()

# create an empty transformation object 
# and 
# use skimage's estimate method to estimate the transformation from the point correspondences:

tform2 = transform.AffineTransform()
tform2.estimate(img_points, img_warped_points)

# We can now apply this transformation to the warped image, 
# creating therefore the aligned version. Let's see how it compares to the original one:

img_aligned = transform.warp(img_warped, tform2)

fig, ax = plt.subplots(1, 3, figsize=(14, 6))
ax[0].imshow(img, cmap='gray'), ax[0].set_title('Original'), ax[0].axis('off')
ax[0].plot(img_points[:, 0], img_points[:, 1], 'xr')
ax[1].imshow(img_warped, cmap='gray'), ax[1].set_title('Warped'), ax[1].axis('off')
ax[1].plot(img_warped_points[:, 0], img_warped_points[:, 1], 'xb')
ax[2].imshow(img_aligned, cmap='gray'), ax[2].set_title('Aligned'), ax[2].axis('off')
fig.tight_layout()
plt.show()


# We can also compare the parameters of both affine transformations directly:
print("Affine transformation parameters for distortion = ", "\n", tform.params)
print("Affine transformation parameters for alignment = ", "\n", tform2.params)

'''
 Image matching with feature descriptors
'''


'''
Watershed segmentation
'''


'''
Tracking through segmentation
'''
print("hellow world")