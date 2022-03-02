'''
Finding Starfish! with k-means

K-Means can be used to segment the image creating a clustering in colour-space.
Let's see how it would work with the starfish image!
'''


from skimage import io, img_as_float,img_as_ubyte
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as npxs

img = io.imread('OnTheBeach.png')

img_reshaped = img_as_float(img).reshape((img.shape[0] * img.shape[1], 3))
print('Img_reshaped shape =', img_reshaped.shape)


n_colours = 6
kmeans = KMeans(n_clusters=n_colours, random_state=0).fit(img_reshaped)
print(kmeans.labels_)

# error here,'NoneType' object has no attribute 'split'???
labels = kmeans.predict(img_reshaped)

img_seg2 = kmeans.cluster_centers_[labels]
img_seg2 = img_as_ubyte(img_seg2.reshape(img.shape))

fig, ax = plt.subplots(ncols=2, figsize=(18, 6))
ax[0].imshow(img), ax[0].axis('off')
ax[1].imshow(img_seg2), ax[1].axis('off')
fig.tight_layout()
plt.show()


# 'NoneType' object has no attribute 'split'
