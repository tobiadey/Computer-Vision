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


cap = cv2.VideoCapture('GregPen.avi')
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
# plt.show()
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
# plt.show()
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

# video shape = (300, 480, 640, 3)
# video[9, :, :, :]  video shape

marker_coord = np.zeros((frameCount, 2))

##############################################################################
# TODO: Track the pink marker                                                #
##############################################################################

from skimage import measure, color

# segmentation based on colour channels
# do frame by frame for loop
fc = 0

while fc < frameCount: 

  # extract cord in for loop for the centroid
  seg_mask = ((video[fc, :, :, 0] > 190) 
            & (video[fc, :, :, 1] > 40)
            & (video[fc, :, :, 1] < 120)
            & (video[fc, :, :, 2] > 120)
            & (video[fc, :, :, 2] < 210))
  print('video shape =', video.shape)
  print('marker coordinates =', marker_coord.shape)
  print('segmask shape =', seg_mask.shape)

  # print(seg_mask)
  # get the x and y value for each frame
  x = video[fc,:,:,:]
  y = video[fc,:,:,:]
 

  # change marker cordinated to the x and y value from above
  # marker_coord[:,0] = x #x channel
  # marker_coord[:,1] = y #y channel

  # # turn it into a variable for ease of use
  x_ch = marker_coord[:,0]
  y_ch = marker_coord[:,1]

  # # check the effect of x and y channel as for loop changes
  # # print('x_ch =',x_ch)
  # # print('y_ch =',y_ch)

  # # for testing purposes
 
  print('x =', x[0])
  print('y =', x[1])

  fc += 1
  print('fc =',fc)

  # use x y in the centroid

  break

  # # use region filtering
  # # apply a filter to remove regions that have incorrect area characteristics.
  # # skimage.measure has a very useful function called regionprops
  # label_img = measure.label(seg_mask) 
  # label_img_rgb = color.label2rgb(label_img, bg_label=0)
  # regions = measure.regionprops(label_img)

  # # dont have to filter, can sort. can get max
  # # area_T = 1000
  # region_ids = [props.label for props in regions.sort() if props.area > area_T] #this is very long i think
  # # label_img_filtered = np.array([px if px in region_ids else 0 for px in label_img.ravel()])


  # look for centriod of first frame

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

# fig, ax = plt.subplots()

# def frame(i):
#     ax.clear()
#     ax.axis('off')
#     ax.plot(marker_coord[i, 1], marker_coord[i, 0], 'og')
#     fig.tight_layout()
#     plot=ax.imshow(video[i, :, :, :])
#     return plot

# anim = animation.FuncAnimation(fig, frame, frames=100)
# plt.close()
# anim