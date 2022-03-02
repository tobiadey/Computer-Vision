'''
Finding Starfish! With thresholding

The goal is to recognise the starfish in the image, and provide an accurate as possible segmentation.
There are many ways to complete this task.

A suggested algorithmic workflow is presented:
Image -> Blur() -> ColourSegmentation()  -> MorphologicalProcessing()  -> RegionFiltering()  -> Result
'''

# load image & display
from skimage import io, img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np


img = io.imread('OnTheBeach.png')

plt.figure(figsize=(9, 6))
plt.imshow(img)
plt.axis('off')
plt.show();

'''
Task 1.1: Blur

Images are corrupted by noise, which can be addressed using a denoising algorithm.

A popular choice for denoising is blurring (a.k.a. low pass filtering),
which reduces abrupt changes in image colour that is characteristic to noise.

Whilst some blurring is often beneficial for image analysis,
performing too much blurring can result in a loss of detail in the image.

blur the image by a small amount.
Feel free to use any type of low pass filter, like a box or Gaussian filter.
'''

##############################################################################
# TODO: Blur the image                                                       #
##############################################################################
from skimage import filters
img_blurred = img_as_ubyte(filters.gaussian(img, sigma=2, multichannel=True))

fig, ax = plt.subplots(1, 2, figsize=(14, 8))
ax[0].imshow(img)
ax[0].set_title('Original'), ax[0].set_axis_off()
ax[1].imshow(img_blurred)
ax[1].set_title('Blurred'), ax[1].set_axis_off()
fig.tight_layout()
plt.show();
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################




'''
Task 1.2: Colour segmentation

In this image, colour is one of the features that distinguishes the starfish from everything else in the image.
As a consequence, colour-based segmentation seems like a very reasonable way forward.

This is like the thresholding we have seen in class, but applied with different values to each colour channel.

The result should be a binary image that has 1 for starfish-coloured pixels,
and 0 for non-starfish coloured pixels.

If you’re unsure how to do this,
remember how you can use the logical operators (e.g. “&”) to combine different binary images.

Segment the image using the following channel-based ranges:
R>155
100<G<180
B<140

Make sure to call the generated binary segmentation map img_seg and display it.
'''

##############################################################################
# TODO: Perform colour-based thresholding                                    #
##############################################################################
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from imageio import imread

# import image
# img = io.imread('OnTheBeach.png')
img = img_blurred

# creation of r g b channels.
print('Image dimensions =', img.shape)
r_ch = img[:, :, 0]
g_ch = img[:, :, 1]
b_ch = img[:, :, 2]


print('Image dimensions rch =', r_ch.shape)


# apply different thersholds values to each colour channel
R = r_ch > 155
G1 = g_ch > 100
G2 = g_ch < 180
B = b_ch < 140

# combine back to img_seg variable
img_seg = R & G1 & G2 & B


plt.figure(figsize=(10, 8))     # Set figure size in inches
plt.imshow(img_seg)
plt.show();
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

'''
Morphological processing
The boundaries of the segmented starfish should be reasonable, but there might be some holes in the starfish segmentations.
This can be easily addressed using morphological processing
scipy.ndimage's function binary_fill_holes can be used to fill holes in a binary image like this one:
'''
from scipy.ndimage import binary_fill_holes

im_filled = binary_fill_holes(img_as_float(img_seg))

plt.figure(figsize=(9, 6))
plt.imshow(im_filled, cmap='gray')
plt.axis('off')
plt.show();



'''
Region filtering

Most likely there is some clutter in your segmentation, with non-starfish objects part of the segmentation mask.
The reason for this is that there are no single thresholds that uniquely identify starfish from non-starfish pixels in this image.
Therefore, some additional processing is required in order to identify the starfish only.

Another feature that distinguishes starfish in the image is shape: starfish have a star shape!
also, starfish are fairly large compared to the smaller shells and “clutter” that appears in the segmentation mask.

So in the last stage of processing
we will apply a filter to remove regions that have incorrect shape characteristics or are too small.

skimage.measure has a very useful function called regionprops that computes basic shape characteristics for regions in image.
'''

# compute the region properties and filter regions to isolate the starfish regions.
# First, we will need to label each separate object in the binary image,
# since this is the type of input that regionprops accepts. This can be done with skimage.measure.label:

from skimage import measure, color

label_img = measure.label(im_filled)
label_img_rgb = color.label2rgb(label_img, bg_label=0)

plt.figure(figsize=(9, 6))
plt.imshow(label_img_rgb)
plt.axis('off')
plt.show();

#regionprops to filter the objects with an area greater than 1000 pixels.
regions = measure.regionprops(label_img)

area_T = 1000
region_ids = [props.label for props in regions if props.area > area_T]

label_img_filtered = np.array([px if px in region_ids else 0 for px in label_img.ravel()])
label_img_filtered = label_img_filtered.reshape(label_img.shape)
label_img_filtered_rgb = color.label2rgb(label_img_filtered, bg_label=0)

plt.figure(figsize=(9, 6))
plt.imshow(label_img_filtered_rgb)
plt.axis('off')
plt.show();


'''
Task 1.3: Improve region filtering

regionprops can estimate other useful metrics.
Try to use some of them to properly segment the starfish in the image and display the obtained result.
'''

##############################################################################
# TODO: Improve region-based filtering                                       #
##############################################################################

regions = measure.regionprops(label_img)

perimeter_T = 265
region_ids = [props.label for props in regions if props.perimeter >perimeter_T]

label_img_filtered = np.array([px if px in region_ids else 0 for px in label_img.ravel()])
label_img_filtered = label_img_filtered.reshape(label_img.shape)
label_img_filtered_rgb = color.label2rgb(label_img_filtered, bg_label=0)

plt.figure(figsize=(9, 6))
plt.imshow(label_img_filtered_rgb)
plt.axis('off')
plt.show();
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################


# Now that we have our final segmentation mask, we can also improve the visualisation using an overlay on the original image:
label_img_filtered_rgb = color.label2rgb(label_img_filtered, image=img, bg_label=0)

plt.figure(figsize=(9, 6))
plt.imshow(label_img_filtered_rgb)
plt.axis('off')
plt.show();
