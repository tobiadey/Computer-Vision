'''
HOG & CNN
'''

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.utils import shuffle
from keras.layers import Conv2D,Conv1D, MaxPooling2D, Dense, Flatten, Dropout


from base import LoadDataHOG

X_train,X_train_HOG,X_train_HOG_flattened,y_train,X_test,X_test_HOG,X_test_HOG_flattened,y_test =LoadDataHOG(100)

print("X_train ",X_train.shape)
print("X_train_HOG ",X_train_HOG.shape)
print("X_train_HOG_flattened ",X_train_HOG_flattened.shape)
print("y_train ", y_train.shape)
print("---------------------------")
print("X_test ",X_test.shape)
print("X_test_HOG ",X_test_HOG.shape)
print("X_test_HOG_flattened ",X_test_HOG_flattened.shape)
print("y_test ", y_test.shape)
# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(10):
#     ax[i].imshow(X_train_HOG[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_train[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout()
# plt.show()

'''
create CNN classifier
'''
# rows = 28
# cols = 28
# shape = (rows, cols, 1)
# filterSize,kernelSize,strides,poolSize,density,epochs,batch_size = 32,3,2,2,128,48,512
#reshape the arrays as needed. apply features scaling.
# X_train = X_train.reshape(X_train.shape[0], *shape)
# X_validate =X_validate.reshape(X_validate.shape[0], *shape)
# X_test = X_test.reshape(X_test.shape[0], *shape)


'''
error here, has to do with shape expected??
'''
classifier = Sequential()

# classifier.add(Conv1D(filters=32, kernel_size=3,padding = 'same', activation='relu'))
classifier.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(25,25)))
classifier.add(Flatten())
classifier.add(Dense(1, activation='sigmoid'))
# Compile the model
classifier.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])# metrics = What you want to maximise

# Train the classifier
history = classifier.fit( X_train_HOG_flattened ,y_train)
# history= classifier.fit(X_train_HOG_flattened, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test_HOG_flattened, y_test))

'''
Preict using CNN Classifier and test data
'''


# predict the classes on the images of the test set
# y_pred = classifier.predict(X_test_HOG_flattened)
score = classifier.evaluate(X_test_HOG_flattened, y_test, verbose=0)
print('test loss: {:.2f}'.format(score[0]))
print('test acc: {:.2f}'.format(score[1]))

