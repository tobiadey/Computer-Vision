'''
Tracking through segmentation

In the previous lab tutorial, we learned a way to segment starfish out of a static image. 

However,Segmentation can be used also to track an object in a video sequence: 
if instead of being given a static image of starfish, 
we were given a video, 
we could apply the segmentation workflow from before to each frame the video, 
providing a pixel-wise tracking of objects through the video sequence. 
Let's see how we can implement this in an example.


First we need to load the video. 
OpenCV provides some functionality to do this: 
however, to better handle the data we will convert it 
to a 4D ndarray of shape [frameCount, frameHeight, frameWidth, nChannels=3]. 
It's important to note that OpenCV loads colour data with BGR coding instead of the more common RGB, 
so we will perform a conversion to avoid surprises.

'''
from skimage.measure import label, regionprops
import cv2
import numpy as np
import matplotlib.pyplot as plt



cap = cv2.VideoCapture('/Users/tobiadewunmi/Desktop/compVis/l5/GregPen.avi')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True

while fc < frameCount and ret:
    ret, video[fc] = cap.read()
    video[fc] = cv2.cvtColor(video[fc], cv2.COLOR_BGR2RGB)
    fc += 1

cap.release()

print('video shape =', video.shape)

# Show the 10th frame
plt.figure(figsize=(9, 6)) 
plt.imshow(video[9, :, :, :])
plt.axis('off')
plt.show()

# Displaying videos in Colab is not straightforward, 
# but using matplotlib.animation submodule will do the trick 
# (have a look at the documentation to better understand how this works).

print("Warning: this will require some time to execute.", "\n")

from matplotlib import rc
import matplotlib.animation as animation
rc('animation', html='jshtml')

fig, ax = plt.subplots()

def frame(i):
    ax.clear()
    ax.axis('off')
    fig.tight_layout()
    plot=ax.imshow(video[i, :, :, :])
    return plot

anim = animation.FuncAnimation(fig, frame, frames=100)
plt.show()
plt.close()
# anim


# using matplotlib.animation for displaying videos 

from matplotlib import rc
import matplotlib.animation as animation
rc('animation', html='jshtml')

fig, ax = plt.subplots()

def frame(i):
    ax.clear()
    ax.axis('off')
    fig.tight_layout()
    plot=ax.imshow(video[i, :, :, :])
    return plot

anim = animation.FuncAnimation(fig, frame, frames=100)
plt.show()
plt.close()
# anim


# now we have loaded our video, 
# converted it into an easy-to-manage format, 
# and we can display it. 
# We have thus all the ingredients for the next task.


'''
Task 4.1: Track the pink marker
Write the code necessary to track the centre of the pink marker in the video sequence.

To do this, you can use a similar approach to the one used for segmenting starfish in the past tutorial. 
In particular, you can use the following channel-based ranges:
𝑅>190 
40<𝐺<120 
120<𝐵<210

Store the marker coordinates in the provided marker_coord array. 
(Tip: to detect the marker, it's advisable to select the region with highest area).
'''

marker_coord = np.zeros((frameCount, 2))


##############################################################################
# TODO: Track the pink marker                                                #
##############################################################################

for i in range(frameCount):
    img_seg = (video[i, :, :, 0] > 190) \
              & (video[i, :, :, 1] > 40) & (video[i, :, :, 1] < 120) \
              & (video[i, :, :, 2] > 120) & (video[i, :, :, 2] < 210)

    label_img = label(img_seg)
    regions = regionprops(label_img)
    sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
    marker_coord[i, :] = sorted_regions[0].centroid

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

fig, ax = plt.subplots()

def frame(i):
    ax.clear()
    ax.axis('off')
    ax.plot(marker_coord[i, 1], marker_coord[i, 0], 'og')
    fig.tight_layout()
    plot=ax.imshow(video[i, :, :, :])
    return plot

anim = animation.FuncAnimation(fig, frame, frames=100)
plt.show()
plt.close()
anim