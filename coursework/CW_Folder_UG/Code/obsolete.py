'''

obsolete 1
'''


# '''
# Intro
# Author Oluwatobi Adewunmi
# loading up the data and preprocessing

# '''

# # import the necessary libraries
# from ast import If
# from cmath import nan
# from fileinput import filename
# from itertools import count
# from lib2to3.pytree import convert
# import math
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from skimage import io, color, img_as_ubyte, img_as_float
# from math import *
# # from ..CW_Dataset.train import * 


# print("Running base.py")
# print("Loading dataset")

# ROOT_DIR = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/CW_Dataset/'

# def convertToFloat(df):
#     # store the while train directory in a variable called files
#     files = os.listdir(ROOT_DIR + 'train')
#     # loop through all files and convert to float, if file is a 
#     count = 1
#     # anotherList = files[:10]
#     anotherList = files
#     # print(anotherList)
#     print("starting")
#     for filename in anotherList:
#         # check if the file is an image, ignore if not an image
#         filename_str = str(filename)
#         # remove the text _alligned theefore it matches the name of file stores in labels
#         filename_str = filename_str.replace('_aligned', '')

#         if (filename.find(".jpg") == -1):
#             print(filename, "ignored")
#             count = count +1
#         else:
#             print(filename_str, count,"/",len(anotherList))
#             image = io.imread(ROOT_DIR + 'train/'+filename)
#             image = img_as_float(image)

#             # change value to new value else keep as old value
#             df['image'] = df['image'].apply(lambda x: image if (x == filename_str) else x )
#             count = count +1
#     print(df)
#     return df
    

# # reference my AI coursework code
# def preProcessData():
#     # Import the dataset
#     df_train = pd.read_csv(ROOT_DIR + 'labels/list_label_train.txt', sep=" ",  header=None, dtype=None)
#     df_test = pd.read_csv(ROOT_DIR + 'labels/list_label_test.txt', sep=" ",  header=None)
#     df_train.columns = ["name", "label"]
#     df_test.columns = ["name", "label"]
#     # print(df_test)

#     # print inormation stored in the dataframes
#     print("-------------------------print head of dataframe (train)------------------------")
#     print(df_train.head())
#     print("-------------------------print head of dataframe (test)------------------------")
#     print(df_test.head())


#     # how many classes am i working with
#     print("-------------------------Check the number of classes(train)------------------------")
#     print(set(df_train['label']))
#     print("-------------------------Check the number of classes(test)------------------------")
#     print(set(df_test['label']))


#     # what are the min and max values for the feature column
#     # how many images are there in total
#     print("-------------------------check minimum and maximum values in the feature variable columns------------------------")

#     print("-------------------------Checking min & max values for (train data)------------------------")
#     print([df_train.drop(labels='label', axis=1).min(axis=1).min(),
#     df_train.drop(labels='label', axis=1).max(axis=1).max()])

#     print("-------------------------Checking min & max values for (test data)------------------------")
#     print([df_test.drop(labels='label', axis=1).min(axis=1).min(),
#     df_test.drop(labels='label', axis=1).max(axis=1).max()])

#     print("-------------------------lets load the image files and convert them to a float of vectors(points)------------------------")
#     # convert each image into a float using img_as_float.
#     image = io.imread(ROOT_DIR + 'train/train_00001_aligned.jpg')
#     image = img_as_float(image)

#     print("-------------------------fit this image float values into the current dataframe and save csv file------------------------")
#     # convert the labal colum to float
#     df_train['label'] = df_train['label'].astype(float)

#     # create new column called image which is a copy of name
#     df_train['image'] = df_train['name']


#     # retrun new df_train that now has the new column updated to the image to float values
#     new_df_train = convertToFloat(df_train)
#     print(new_df_train)

#     # store this dataframe as a csv in order for ease of use later, as it takes a while to create
#     new_df_train.to_csv(ROOT_DIR +'/updated.csv', sep=';')
#     print('check for csv')
#     # test that the values can produce an image by checking the first value in the image column
#     first_image = new_df_train['image'].iat[0]
#     plt.imshow(first_image)
#     plt.show()


#     '''
#     convert each image into a float
#     but how they are stored in a different place
#     I have the image name not the images themselves??
#     '''
#     #convert data from unsigned integers to float.
#     # train_data = np.array(df_train,dtype='float32')
#     # test_data = np.array(df_test,dtype='float32')



#     # Features scaling
#     # Pixel data (divided by 255 to rescale to 0-1 and not 0-255)
#     # As the target variable is the labels column(1st column), this will be our y variable.

#     # # training data
#     # x_train = train_data[:,1:]/255
#     # y_train = train_data[:,0]

#     # # testing data
#     # x_test = test_data[:,1:]/255
#     # y_test = test_data[:,0]

#     # #print the shape of X and y
#     # print('\n-------------------------training and testing data shapes------------------\n')
#     # print('X: ' + str(x_train.shape))
#     # print('Y:  '+ str( y_train.shape))
#     # print('X_test: ' + str(x_test.shape))
#     # print('Y_test:  '+ str( y_test.shape))


#     # return x_train, y_train,x_test,y_test

# def processData():
#      # Import the dataset
#     train_data = np.genfromtxt(ROOT_DIR +'/updated.csv', delimiter='')
#     print(train_data.shape) #train data split into FileName(img) and classification value

#     # X_train = train_data[:, 1:]
#     # y_train = train_data[:, 1] #[5. 5. 4. ... 7. 7. 7.]

#     # print(X_train)
#     # print(y_train)



# # X, y,X_test,y_test = processData()

# '''
# Main body 
# '''


# # check for file that shows the new csv has been created
# file_exists = os.path.exists(ROOT_DIR + '/updated.csv')

# # if this file exists no need to create a new csv therefore go stright into working with the new csv
# if file_exists:
#     processData()
# else:
#     preProcessData()
#     processData()



# print("End of base.py")
# print("\n")


'''
obsolete 2
'''

# '''
# Intro
# Author Oluwatobi Adewunmi
# loading up the data and preprocessing

# '''

# # import the necessary libraries
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io, color, img_as_ubyte, img_as_float
# # from sklearn.model_selection import train_test_split
# from PIL import Image
# from keras.preprocessing.image import load_img, img_to_array

# print("Running base.py")
# print("Loading dataset")
# ROOT_DIR = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/CW_Dataset/'

# def processImages(train_data):
#     # store all files  train directory in a variable called files
#     files = os.listdir(ROOT_DIR + 'train')

#     # loop through all files and convert to float, if file is a 
#     count = 0
#     anotherList = files
#     for filename in anotherList:
#         # check if the file is an image, ignore if not an image
#         filename_str = str(filename)
#         # remove the text _alligned theefore it matches the name of file stores in labels
#         filename_str = filename_str.replace('_aligned', '')
#         if (filename.find(".jpg") == -1):
#             print(filename, "ignored")
#             count = count +1
#         else:
#             print(filename_str, count,"/",len(anotherList))
#             # convert each image into a float using img_as_float.
#             image = io.imread(ROOT_DIR + 'train/'+filename)
#             image = img_as_float(image)
#             # convert image of float to numpy array
#             img_array = img_to_array(image)

#             for x in train_data: 
#                 # print("x value", x)
#                 # print("utf-", x[0].decode('UTF-8'))
                
#                 # before dcoding check it is an image file name or has already been chcnaged to 'numpy.ndarray' object
#                 # print(type(x[0]))
#                 # print(x[0])

#                 if type(x[0]) is np.ndarray:
#                     stringVal = "name" 
#                     # print("numpy.ndarray, not image")
#                 else:
#                     stringVal = x[0].decode('UTF-8')
#                     # print("image")

#                 if(stringVal == filename_str):
#                     x[0]= img_array
#         count = count +1

#     return train_data

# def preProcessData():
#     # Import the dataset
#     train_data = np.genfromtxt(ROOT_DIR + 'labels/list_label_train.txt', delimiter=" ", dtype=object)
#     print(train_data.shape) #train data split into FileName(img) and classification value
#     # (12271,)

#     train_data = train_data[:5]
#     print("pre processing ", train_data)
#     train_data = processImages(train_data)
#     # print("post processing ", train_data)

#     # print(train_data[0])
#     # print(train_data[0][0])

#     print(train_data.shape) #(5,)
#     train_data = np.reshape(train_data, (-1, 2))
#     print(train_data.shape)

#     # write an array of object to csv by using a formatter '%s' (string)
#     # np.savetxt(ROOT_DIR +'/updated.csv', [train_data], delimiter=",", fmt ='%d')

# def processData():
#     train_data = np.genfromtxt(ROOT_DIR + '/updated.csv', delimiter=",")
#     print(train_data.shape)
#     X_train = train_data[:, 1:]
#     y_train = train_data[:, 0]

#     # first_image = train_data[0][0]
#     # plt.imshow(first_image)
#     # plt.show()
    
#     # print(X_train)
#     # print((y_train))




#     # # Rescale the intensities, cast the labels to int
#     # X_train = X_train / 255.
#     # y_train = y_train.astype(int)

#     # print('X_train shape =', X_train.shape)
#     # print('y_train shape =', y_train.shape)

#     # # reduce training set samples to enable faster computing (but potentially lowering the accuracy of our model):
#     # n_train_samples = 2000
#     # X_train = X_train[:n_train_samples]  #take the first 2000
#     # y_train = y_train[:n_train_samples]

#     # print(X_train)
#     # print(y_train)

# # check for file that shows the new csv has been created
# file_exists = os.path.exists(ROOT_DIR + '/updated.csv')

# # if this file exists no need to create a new csv therefore go stright into working with the new csv
# if file_exists:
#     # work with updated.csv
#     preProcessData() #remove this later
#     processData()
# else:
#     # create updated.csv then work with it
#     preProcessData()
#     processData()



# # Split into train and test
# # feature detection
# # Create a classifier: a support vector classifier
# # Train the classifier


'''
obsolete 3
'''

# # import the necessary libraries
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io, color, img_as_ubyte, img_as_float,data, exposure
# from skimage.feature import hog
# import tensorflow as tf
# import cv2
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
# import pandas as pd
# from tqdm import tqdm
# from sklearn import svm, metrics
# from sklearn.utils import shuffle



# print("Running base.py")
# print("Loading dataset")
# DATASET_DIR = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/CW_Dataset/'


# '''
# Load train data 
# '''

# # load the train data
# df_train = pd.read_csv(DATASET_DIR + '/labels/list_label_train.txt', sep=" ",  header=None, dtype=None)

# # name the colums
# df_train.columns = ["name", "label"]

# # store the image array an label in a 1d array 
# train_image = []
# train_image_HOG = []
# train_image_data = []
# testSize = df_train.shape[0]


# # shorten the train data for testing
# df_train = df_train[:testSize]
# print(df_train)


# # loop through the csv files, using the file name to laod the image and store in trian_image array
# for i in tqdm(range(df_train.shape[0])):

#     # get the file name,remove the .jpg and store in val
#     val = df_train['name'][i]
#     val = val.split('.')[0] 

#     # get the target value assiciated with the filename
#     target = df_train['label'][i]
#     # print(val, ":", target)

#     # load image based on the filename colelcted from the csv
#     img = image.load_img(DATASET_DIR +'/train/'+val+'_aligned.jpg', target_size=(100,100,1))

#     # apply HOG feature descriptor
#     HOG_des, HOG_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)

#     # # Rescale histogram for better display
#     HOG_image_rescaled = exposure.rescale_intensity(HOG_image, in_range=(0, 10))

#     # convert the given image into  numpy array
#     img = image.img_to_array(img) #print(type(img_numpy_array))
#     # Rescale the intensities
#     img = img/255
#     train_image.append(img)
#     train_image_HOG.append(HOG_image_rescaled)
#     train_image_data.append(target)

# X_train = np.array(train_image)
# X_train_HOG = np.array(train_image_HOG)
# y_train = np.array(train_image_data)
# print("X_train ",X_train.shape)
# print("X_train_HOG ",X_train_HOG.shape)
# print("y_train ", y_train.shape)
# # print(X_train[0])
# # print(X_train_HOG[0])
# # print(y_train[0])



# # plt.imshow(X_train[0])
# # plt.imshow(X_train_HOG[0])
# # plt.show()

# # fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), sharex=True, sharey=True)
# # ax = axes.ravel()

# # for i in range(10):
# #     ax[i].imshow(X_train[i], cmap='gray')
# #     ax[i].set_title(f'Label: {y_train[i]}')
# #     ax[i].set_axis_off()
# # fig.tight_layout()
# # plt.show()


# '''
# create classifier
# '''

# # from train_SVM import train_linear_SVM
# # Create a classifier: a support vector classifier
# # classifier = svm.SVC(gamma=0.001)
# classifier = svm.SVC(kernel='linear')

# print('X_train shape =', X_train.shape)
# print('X_train_HOG shape =', X_train_HOG.shape)
# print('y_train shape =', y_train.shape)
# # X_train shape = (20000, 784)
# # y_train shape = (20000,)

# X_train_flattened = X_train.reshape(len(X_train),-1)
# X_train_HOG_flattened = X_train_HOG.reshape(len(X_train),-1)
# print('X_flatenned =', X_train_flattened.shape)
# print('X_HOG_flatenned =', X_train_HOG_flattened.shape)

# # Train the classifier
# classifier.fit( X_train_HOG_flattened ,y_train)


# '''
# load test data
# '''

# # Now it's time to check how this classifier performs on the test set. Let's load and rescale it first:
# # load the test data
# df_test = pd.read_csv(DATASET_DIR + '/labels/list_label_test.txt', sep=" ",  header=None, dtype=None)

# # name the colums
# df_test.columns = ["name", "label"]

# # store the image array an label in array 
# test_image = []
# test_image_HOG = []
# test_image_data = []

# # shorten the train data for testing
# df_test = df_test[:testSize]
# print(df_test)

# # loop through the csv files, using the file name to laod the image and store in test_image array
# for i in tqdm(range(df_test.shape[0])):

#     # get the file name,remove the .jpg and store in val
#     val = df_test['name'][i]
#     val = val.split('.')[0] 

#     # get the target value assiciated with the filename
#     target = df_test['label'][i]
#     # print(val, ":", target)

#     # load image based on the filename colelcted from the csv
#     img = image.load_img(DATASET_DIR +'/test/'+val+'_aligned.jpg', target_size=(100,100,1))

#     # apply HOG feature descriptor
#     HOG_des, HOG_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)

#     # # Rescale histogram for better display
#     HOG_image_rescaled = exposure.rescale_intensity(HOG_image, in_range=(0, 10))

#     # convert the given image into  numpy array
#     img = image.img_to_array(img) #print(type(img_numpy_array))
#     # Rescale the intensities
#     img = img/255
#     test_image.append(img)
#     test_image_data.append(target)
#     test_image_HOG.append(HOG_image_rescaled)

# X_test = np.array(test_image)
# X_test_HOG = np.array(test_image_HOG)
# y_test = np.array(test_image_data)
# print(X_test.shape)
# print(X_test_HOG.shape)
# print(y_test.shape)
# # # print(X_test[0])
# # # print(y_test[0])
# # plt.imshow(X_test[0])
# # plt.show()

# X_test_flattened= X_test.reshape(len(X_test),-1)
# X_test_HOG_flattened= X_test_HOG.reshape(len(X_test_HOG),-1)
# print('X_flatenned =', X_test_flattened.shape)
# print('X_HOG_flatenned =', X_test_HOG_flattened.shape)

# # predict the classes on the images of the test set
# y_pred = classifier.predict(X_test_HOG_flattened)

# X_test, y_test, y_pred = shuffle(X_test, y_test, y_pred)
# X_test_img = X_test

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
# ax = axes.ravel()

# for i in range(10):
#     ax[i].imshow(X_test_img[i], cmap='gray')
#     ax[i].set_title(f'Label: {y_test[i]}\n Prediction: {y_pred[i]}')
#     ax[i].set_axis_off()
# fig.tight_layout
# plt.show()

# print(f"""Classification report for classifier {classifier}:\n
#       {metrics.classification_report(y_test, y_pred)}""")