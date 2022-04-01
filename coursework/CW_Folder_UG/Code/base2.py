'''
Intro
Author Oluwatobi Adewunmi
loading up the data and preprocessing

'''

# import the necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte, img_as_float
# from sklearn.model_selection import train_test_split
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array

print("Running base.py")
print("Loading dataset")
ROOT_DIR = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/CW_Dataset/'

def processImages(train_data):
    # store all files  train directory in a variable called files
    files = os.listdir(ROOT_DIR + 'train')

    # loop through all files and convert to float, if file is a 
    count = 0
    anotherList = files
    for filename in anotherList:
        # check if the file is an image, ignore if not an image
        filename_str = str(filename)
        # remove the text _alligned theefore it matches the name of file stores in labels
        filename_str = filename_str.replace('_aligned', '')
        if (filename.find(".jpg") == -1):
            print(filename, "ignored")
            count = count +1
        else:
            print(filename_str, count,"/",len(anotherList))
            # convert each image into a float using img_as_float.
            image = io.imread(ROOT_DIR + 'train/'+filename)
            image = img_as_float(image)
            # convert image of float to numpy array
            img_array = img_to_array(image)

            for x in train_data: 
                # print("x value", x)
                # print("utf-", x[0].decode('UTF-8'))
                
                # before dcoding check it is an image file name or has already been chcnaged to 'numpy.ndarray' object
                # print(type(x[0]))
                # print(x[0])

                if type(x[0]) is np.ndarray:
                    stringVal = "name" 
                    # print("numpy.ndarray, not image")
                else:
                    stringVal = x[0].decode('UTF-8')
                    # print("image")

                if(stringVal == filename_str):
                    x[0]= img_array
        count = count +1

    return train_data

def preProcessData():
    # Import the dataset
    train_data = np.genfromtxt(ROOT_DIR + 'labels/list_label_train.txt', delimiter=" ", dtype=object)
    print(train_data.shape) #train data split into FileName(img) and classification value
    # (12271,)

    train_data = train_data[:5]
    print("pre processing ", train_data)
    train_data = processImages(train_data)
    # print("post processing ", train_data)

    # print(train_data[0])
    # print(train_data[0][0])

    print(train_data.shape) #(5,)
    train_data = np.reshape(train_data, (-1, 2))
    print(train_data.shape)

    # write an array of object to csv by using a formatter '%s' (string)
    # np.savetxt(ROOT_DIR +'/updated.csv', [train_data], delimiter=",", fmt ='%d')

def processData():
    train_data = np.genfromtxt(ROOT_DIR + '/updated.csv', delimiter=",")
    print(train_data.shape)
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]

    # first_image = train_data[0][0]
    # plt.imshow(first_image)
    # plt.show()
    
    # print(X_train)
    # print((y_train))




    # # Rescale the intensities, cast the labels to int
    # X_train = X_train / 255.
    # y_train = y_train.astype(int)

    # print('X_train shape =', X_train.shape)
    # print('y_train shape =', y_train.shape)

    # # reduce training set samples to enable faster computing (but potentially lowering the accuracy of our model):
    # n_train_samples = 2000
    # X_train = X_train[:n_train_samples]  #take the first 2000
    # y_train = y_train[:n_train_samples]

    # print(X_train)
    # print(y_train)

# check for file that shows the new csv has been created
file_exists = os.path.exists(ROOT_DIR + '/updated.csv')

# if this file exists no need to create a new csv therefore go stright into working with the new csv
if file_exists:
    # work with updated.csv
    preProcessData() #remove this later
    processData()
else:
    # create updated.csv then work with it
    preProcessData()
    processData()



# Split into train and test
# feature detection
# Create a classifier: a support vector classifier
# Train the classifier