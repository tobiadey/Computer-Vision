'''
Image matching with feature descriptors

manually picking point correpondences in images is not a very convenient approach for image matching, 
especially if we are dealing with more than just a few images.

we can perform the same task by means of interest point detection and feature descriptors. 
This approach is essentially based on the following steps:
- Automatically detect interest points (or keypoints) in both images independently
- Compute feature descriptors for each of these points
- Find point correspondences between the two sets of points (by comparing their feature descriptors
- Estimate the transformation (just as we did with manually-selected correspondences)
'''

# First, let's load again our cameraman image 
# (note that we will need to cast it to uint8 in order to use OpenCV's functions) 
# and apply a warping to corrupt it:

from skimage import data, img_as_ubyte, transform
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = img_as_ubyte(data.camera())  # Format required by OpenCV

# warp synthetic image (default: scale=(0.9, 0.9), rotation=0.2, translation=(20, -10))
tform = transform.AffineTransform(scale=(0.9, 0.9), rotation=0.2, translation=(20, -10))
img_warped = img_as_ubyte(transform.warp(img, tform.inverse))

fig, ax = plt.subplots(1, 2, figsize=(9, 6))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original'), ax[0].axis('off')
ax[1].imshow(img_warped, cmap='gray')
ax[1].set_title('Warped'), ax[1].axis('off')
fig.tight_layout()
plt.show()


# To identify interest points and compute feature descriptors, 
# we will make use of SIFT, 
# which is arguably the most famous approach for this task. 

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Identify the keypoints and compute the descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(img_warped, None)

# Now it's time to perform the matching: 
# to do this, OpenCV provides the convenient BFMatcher object, 
# where we can specify also the type of metric used to perform the comparisons. 
# Make sure to have a look at the documentation online to understand how to use this object and its methods.

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)


# Then, we can sort the matches between the two sets of keypoints, 
# and keep only a portion of them (trying to discard spurious and unreliable matches):
# Sort matches based on differences between matched descriptors
matches = sorted(matches, key=lambda x: x.distance)

# Remove bottom-half of the matches
good_match_ratio = 0.5
numGoodMatches = int(len(matches) * good_match_ratio)
matches = matches[:numGoodMatches]

# OpenCV provides a very handy function to also plot the matches between the two sets of points:
# Show matches
fig = plt.figure(figsize=(12, 9))
imMatches = cv2.drawMatches(img, kp1, img_warped, kp2, matches, None)
plt.imshow(imMatches)
plt.show()

# Notice the amount of keypoints that are being matched: 
# the more reliable matches, the better to accurately estimate the final transformation.

# Now, using the matching information, 
# we can create two ordered arrays with the point coordinates for the two sets of points. 
# Specifically, the rows of points1 and of points2 will consist of the  (𝑥,𝑦)  coordinates of matched points, 
# just as we did in the previous section. 
# This is needed to estimate the alignment transformation.

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

# We can now use OpenCV's findHomography function to estimate the transformation parameters. 
# This function allows to also use RANSAC (of which we talked about in the lecture) 
# to provide a more robust estimation. 
# Finally, we can use the estimated transformation to generate the aligned image from the warped one:

inv_h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

img_aligned = transform.warp(img_warped, inv_h)

fig, ax = plt.subplots(1, 3, figsize=(14, 6))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original'), ax[0].axis('off')
ax[1].imshow(img_warped, cmap='gray')
ax[1].set_title('Warped'), ax[1].axis('off')
ax[2].imshow(img_aligned, cmap='gray')
ax[2].set_title('Aligned'), ax[1].axis('off')
fig.tight_layout()
plt.show()


'''
Task 2.1: Break the algorithm
Try other, more dramatic distortions (get creative!). Can you break the image matching algorithm we just implemented?
'''