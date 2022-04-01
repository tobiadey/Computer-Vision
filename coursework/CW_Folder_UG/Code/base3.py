# import the necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte, img_as_float,data, exposure
from skimage.feature import hog
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import pandas as pd
from tqdm import tqdm
from sklearn import svm, metrics
from sklearn.utils import shuffle



print("Running base.py")
print("Loading dataset")
DATASET_DIR = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/CW_Dataset/'


'''
Load train data 
'''

# load the train data
df_train = pd.read_csv(DATASET_DIR + '/labels/list_label_train.txt', sep=" ",  header=None, dtype=None)

# name the colums
df_train.columns = ["name", "label"]

# store the image array an label in a 1d array 
train_image = []
train_image_HOG = []
train_image_data = []
testSize = df_train.shape[0]


# shorten the train data for testing
df_train = df_train[:testSize]
print(df_train)


# loop through the csv files, using the file name to laod the image and store in trian_image array
for i in tqdm(range(df_train.shape[0])):

    # get the file name,remove the .jpg and store in val
    val = df_train['name'][i]
    val = val.split('.')[0] 

    # get the target value assiciated with the filename
    target = df_train['label'][i]
    # print(val, ":", target)

    # load image based on the filename colelcted from the csv
    img = image.load_img(DATASET_DIR +'/train/'+val+'_aligned.jpg', target_size=(100,100,1))

    # apply HOG feature descriptor
    HOG_des, HOG_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

    # # Rescale histogram for better display
    HOG_image_rescaled = exposure.rescale_intensity(HOG_image, in_range=(0, 10))

    # convert the given image into  numpy array
    img = image.img_to_array(img) #print(type(img_numpy_array))
    # Rescale the intensities
    img = img/255
    train_image.append(img)
    train_image_HOG.append(HOG_image_rescaled)
    train_image_data.append(target)

X_train = np.array(train_image)
X_train_HOG = np.array(train_image_HOG)
y_train = np.array(train_image_data)
print("X_train ",X_train.shape)
print("X_train_HOG ",X_train_HOG.shape)
print("y_train ", y_train.shape)
# print(X_train[0])
# print(X_train_HOG[0])
# print(y_train[0])



# plt.imshow(X_train[0])
# plt.imshow(X_train_HOG[0])
# plt.show()

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(10):
#     ax[i].imshow(X_train[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_train[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout()
# plt.show()


'''
create classifier
'''

# from train_SVM import train_linear_SVM
# Create a classifier: a support vector classifier
# classifier = svm.SVC(gamma=0.001)
classifier = svm.SVC(kernel='linear')

print('X_train shape =', X_train.shape)
print('X_train_HOG shape =', X_train_HOG.shape)
print('y_train shape =', y_train.shape)
# X_train shape = (20000, 784)
# y_train shape = (20000,)

X_train_flattened = X_train.reshape(len(X_train),-1)
X_train_HOG_flattened = X_train_HOG.reshape(len(X_train),-1)
print('X_flatenned =', X_train_flattened.shape)
print('X_HOG_flatenned =', X_train_HOG_flattened.shape)

# Train the classifier
classifier.fit( X_train_HOG_flattened ,y_train)


'''
load test data
'''

# Now it's time to check how this classifier performs on the test set. Let's load and rescale it first:
# load the test data
df_test = pd.read_csv(DATASET_DIR + '/labels/list_label_test.txt', sep=" ",  header=None, dtype=None)

# name the colums
df_test.columns = ["name", "label"]

# store the image array an label in array 
test_image = []
test_image_HOG = []
test_image_data = []

# shorten the train data for testing
df_test = df_test[:testSize]
print(df_test)

# loop through the csv files, using the file name to laod the image and store in test_image array
for i in tqdm(range(df_test.shape[0])):

    # get the file name,remove the .jpg and store in val
    val = df_test['name'][i]
    val = val.split('.')[0] 

    # get the target value assiciated with the filename
    target = df_test['label'][i]
    # print(val, ":", target)

    # load image based on the filename colelcted from the csv
    img = image.load_img(DATASET_DIR +'/test/'+val+'_aligned.jpg', target_size=(100,100,1))

    # apply HOG feature descriptor
    HOG_des, HOG_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

    # # Rescale histogram for better display
    HOG_image_rescaled = exposure.rescale_intensity(HOG_image, in_range=(0, 10))

    # convert the given image into  numpy array
    img = image.img_to_array(img) #print(type(img_numpy_array))
    # Rescale the intensities
    img = img/255
    test_image.append(img)
    test_image_data.append(target)
    test_image_HOG.append(HOG_image_rescaled)

X_test = np.array(test_image)
X_test_HOG = np.array(test_image_HOG)
y_test = np.array(test_image_data)
print(X_test.shape)
print(X_test_HOG.shape)
print(y_test.shape)
# # print(X_test[0])
# # print(y_test[0])
# plt.imshow(X_test[0])
# plt.show()

X_test_flattened= X_test.reshape(len(X_test),-1)
X_test_HOG_flattened= X_test_HOG.reshape(len(X_test_HOG),-1)
print('X_flatenned =', X_test_flattened.shape)
print('X_HOG_flatenned =', X_test_HOG_flattened.shape)

# predict the classes on the images of the test set
y_pred = classifier.predict(X_test_HOG_flattened)

X_test, y_test, y_pred = shuffle(X_test, y_test, y_pred)
X_test_img = X_test

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    ax[i].imshow(X_test_img[i], cmap='gray')
    ax[i].set_title(f'Label: {y_test[i]}\n Prediction: {y_pred[i]}')
    ax[i].set_axis_off()
fig.tight_layout
plt.show()

print(f"""Classification report for classifier {classifier}:\n
      {metrics.classification_report(y_test, y_pred)}""")