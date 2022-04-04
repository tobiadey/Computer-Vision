'''
HOG & SVM
'''

# import the necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte, img_as_float,data, exposure
from skimage.feature import hog
import tensorflow as tf
import cv2
from keras.preprocessing import image
import pandas as pd
from tqdm import tqdm
from sklearn import svm, metrics
from sklearn.utils import shuffle



from base import LoadData

X_train,X_train_flattened,y_train,X_test, X_test_flattened,y_test = LoadData(100)

print("X_train ",X_train.shape)
print("X_train_flattened ",X_train_flattened.shape)
print("y_train ", y_train.shape)
print("---------------------------")
print("X_test ",X_test.shape)
print("X_test_flattened ",X_test_flattened.shape)
print("y_test ", y_test.shape)

'''
create CNN classifier
'''
