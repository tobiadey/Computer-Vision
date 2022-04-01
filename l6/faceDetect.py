'''
test a cascade classifier for face detection 
and use it to automate a task that we performed manually in a previous lab 
(i.e. blurring the faces in a class photo)

we have introduced the famous Viola-Jones object detector framework, 
which was originally used for face detection. 
OpenCV has a specific object for this framework called CascadeClassifier, 
that also comes with several pre-trained models.

'''

from skimage import io, color, img_as_ubyte
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches

img = io.imread('ClassLarge.jpg')
img_gray = color.rgb2gray(img)
img_gray = img_as_ubyte(img_gray)         # Conversion required by OpenCV

fig, ax = plt.subplots(figsize=(18, 12))
ax.imshow(img), ax.set_axis_off()
fig.tight_layout
plt.show()

# Now let's load the haarcascade_frontalface_default.xml model into a CascadeClassifier object 
# and test it on the image. 
# Make sure to understand both the input arguments and the generated output for the detectMultiScale method 
# by looking at the online documentation.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
print('faces shape =', faces.shape)

# In particular, 
# faces is an array of coordinates of bounding boxes relative to the faces detected in the image. 
# Based on its shape, we can say that the detector has identified 31 faces in the image. 
# Let's plot them using matplotlib.patches:

fig, ax = plt.subplots(figsize=(18, 12))
ax.imshow(img), ax.set_axis_off()

for face in faces:
    ax.add_patch(
        patches.Rectangle(xy=(face[0], face[1]), width=face[2], height=face[3],
                          fill=False, color='r', linewidth=2))
fig.tight_layout
plt.show()


##############################################################################
# TODO: Automatically blur faces                                             #
##############################################################################
from skimage import filters

img_blurred = img_as_ubyte(filters.gaussian(img, sigma=13, multichannel=True))
img_composite = img.copy()

for face in faces:
    img_composite[face[1]:face[1]+face[3], face[0]:face[0]+face[2], :] = \
        img_blurred[face[1]:face[1]+face[3], face[0]:face[0]+face[2], :]

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

fig, ax = plt.subplots(figsize=(18, 12))
ax.imshow(img_composite), ax.set_axis_off()
fig.tight_layout
plt.show()


##############################################################################
# TODO: Break the detector
##############################################################################
from skimage import transform

img = img_as_ubyte(transform.rescale(img, 0.25, multichannel=True, anti_aliasing=True))
img_gray = color.rgb2gray(img)
img_gray = img_as_ubyte(img_gray)         # Conversion required by OpenCV

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

fig, ax = plt.subplots(figsize=(18, 12))
ax.imshow(img), ax.set_axis_off()

for face in faces:
    ax.add_patch(
        patches.Rectangle(xy=(face[0], face[1]), width=face[2], height=face[3],
                          fill=False, color='r', linewidth=2)
    )
fig.tight_layout
plt.show()