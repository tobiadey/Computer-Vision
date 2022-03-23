'''
extract histograms of gradients (HOGs) features from an image.

 the histograms of gradients (HOGs) feature descriptor. 
 This descriptor can be extremely powerful to distinguish different shapes in images.

 Let's learn how to extract HOGs from a sample image using the skimage.feature.hog function:
'''

from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt

image = data.astronaut()

HOG_des, HOG_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

# The HOG descriptor is stored in the HOG_des variable, 
# which has the following number of dimensions for the given settings:
print('HOG descriptor shape =', HOG_des.shape)

# If skimage.feature.hog is called with visualize=True, 
# then a visualisation of the feature descriptor is also created (here called HOG_image):

fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

ax[0].axis('off')
ax[0].imshow(image)
ax[0].set_title('Input image')

# Rescale histogram for better display
HOG_image_rescaled = exposure.rescale_intensity(HOG_image, in_range=(0, 10))

ax[1].axis('off')
ax[1].imshow(HOG_image_rescaled, cmap='gray')
ax[1].set_title('Histogram of Oriented Gradients')
fig.tight_layout()
plt.show()

# Note that this image is generated only to better understand 
# what is the information stored in the HOG descriptor.