'''
Intro
Author Oluwatobi Adewunmi
loading up the data and preprocessing

'''

# import the necessary libraries
from ast import If
from cmath import nan
from fileinput import filename
from itertools import count
from lib2to3.pytree import convert
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage import io, color, img_as_ubyte, img_as_float
from math import *
# from ..CW_Dataset.train import * 


print("Running base.py")
print("Loading dataset")

ROOT_DIR = '/Users/tobiadewunmi/Desktop/compVis/coursework/CW_Folder_UG/CW_Dataset/'

def convertToFloat(df):
    # store the while train directory in a variable called files
    files = os.listdir(ROOT_DIR + 'train')
    # loop through all files and convert to float, if file is a 
    count = 1
    # anotherList = files[:10]
    anotherList = files
    # print(anotherList)
    print("starting")
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
            image = io.imread(ROOT_DIR + 'train/'+filename)
            image = img_as_float(image)

            # change value to new value else keep as old value
            df['image'] = df['image'].apply(lambda x: image if (x == filename_str) else x )
            count = count +1
    print(df)
    return df
    

# reference my AI coursework code
def preProcessData():
    # Import the dataset
    df_train = pd.read_csv(ROOT_DIR + 'labels/list_label_train.txt', sep=" ",  header=None, dtype=None)
    df_test = pd.read_csv(ROOT_DIR + 'labels/list_label_test.txt', sep=" ",  header=None)
    df_train.columns = ["name", "label"]
    df_test.columns = ["name", "label"]
    # print(df_test)

    # print inormation stored in the dataframes
    print("-------------------------print head of dataframe (train)------------------------")
    print(df_train.head())
    print("-------------------------print head of dataframe (test)------------------------")
    print(df_test.head())


    # how many classes am i working with
    print("-------------------------Check the number of classes(train)------------------------")
    print(set(df_train['label']))
    print("-------------------------Check the number of classes(test)------------------------")
    print(set(df_test['label']))


    # what are the min and max values for the feature column
    # how many images are there in total
    print("-------------------------check minimum and maximum values in the feature variable columns------------------------")

    print("-------------------------Checking min & max values for (train data)------------------------")
    print([df_train.drop(labels='label', axis=1).min(axis=1).min(),
    df_train.drop(labels='label', axis=1).max(axis=1).max()])

    print("-------------------------Checking min & max values for (test data)------------------------")
    print([df_test.drop(labels='label', axis=1).min(axis=1).min(),
    df_test.drop(labels='label', axis=1).max(axis=1).max()])

    print("-------------------------lets load the image files and convert them to a float of vectors(points)------------------------")
    # convert each image into a float using img_as_float.
    image = io.imread(ROOT_DIR + 'train/train_00001_aligned.jpg')
    image = img_as_float(image)

    print("-------------------------fit this image float values into the current dataframe and save csv file------------------------")
    # convert the labal colum to float
    df_train['label'] = df_train['label'].astype(float)

    # create new column called image which is a copy of name
    df_train['image'] = df_train['name']


    # retrun new df_train that now has the new column updated to the image to float values
    new_df_train = convertToFloat(df_train)
    print(new_df_train)

    # store this dataframe as a csv in order for ease of use later, as it takes a while to create
    new_df_train.to_csv(ROOT_DIR +'/updated.csv', sep=';')
    print('check for csv')
    # test that the values can produce an image by checking the first value in the image column
    first_image = new_df_train['image'].iat[0]
    plt.imshow(first_image)
    plt.show()


    '''
    convert each image into a float
    but how they are stored in a different place
    I have the image name not the images themselves??
    '''
    #convert data from unsigned integers to float.
    # train_data = np.array(df_train,dtype='float32')
    # test_data = np.array(df_test,dtype='float32')



    # Features scaling
    # Pixel data (divided by 255 to rescale to 0-1 and not 0-255)
    # As the target variable is the labels column(1st column), this will be our y variable.

    # # training data
    # x_train = train_data[:,1:]/255
    # y_train = train_data[:,0]

    # # testing data
    # x_test = test_data[:,1:]/255
    # y_test = test_data[:,0]

    # #print the shape of X and y
    # print('\n-------------------------training and testing data shapes------------------\n')
    # print('X: ' + str(x_train.shape))
    # print('Y:  '+ str( y_train.shape))
    # print('X_test: ' + str(x_test.shape))
    # print('Y_test:  '+ str( y_test.shape))


    # return x_train, y_train,x_test,y_test

def processData():
     # Import the dataset
    train_data = np.genfromtxt(ROOT_DIR +'/updated.csv', delimiter='')
    print(train_data.shape) #train data split into FileName(img) and classification value

    # X_train = train_data[:, 1:]
    # y_train = train_data[:, 1] #[5. 5. 4. ... 7. 7. 7.]

    # print(X_train)
    # print(y_train)



# X, y,X_test,y_test = processData()

'''
Main body 
'''


# check for file that shows the new csv has been created
file_exists = os.path.exists(ROOT_DIR + '/updated.csv')

# if this file exists no need to create a new csv therefore go stright into working with the new csv
if file_exists:
    processData()
else:
    preProcessData()
    processData()



print("End of base.py")
print("\n")