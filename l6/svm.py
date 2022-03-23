'''
Digit classification with Support Vector Machines (SVMs)
support vector machines are a very powerful classification algorithm that can be used for image classification.
we will attempt to train an SVM on the MNIST dataset of digits and see how it performs.

we will use simple pixel intensities as feature descriptors: 
since each image in MNIST is 28x28 pixels, 
we will therefore use a 784-dimensional feature descriptors. 
This is not perhaps the smartest choice, but it's worth using it as a starting point.

Let's start by loading the training data and labels in two separate arrays 
and converting the pixel values to float. 
Note that in this case we will use numpy's astype method 
instead of skimage functions because the data is not encoded into actual image files.

'''

# from Code.train_SVM import x
# print(x)

from sklearn import svm, metrics
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np


train_data = np.genfromtxt('/Users/tobiadewunmi/Desktop/compVis/l6/sample_data/mnist_train_small.csv', delimiter=";")
print(train_data.shape) #why does this give nan nan nan when delimiter is , should be a ", rather than ";'

X_train = train_data[:,1:]
y_train = train_data[:, 0]

# Rescale the intensities, cast the labels to int
X_train = X_train / 255.
y_train = y_train.astype(int)

print('X_train shape =', X_train.shape)
print('y_train shape =', y_train.shape)


# As you can see, we have 20001 samples in the training set. \
# Let's reduce this a bit to enable faster computing (but potentially lowering the accuracy of our model):
n_train_samples = 2000
X_train = X_train[:n_train_samples]  #take the first 2000
y_train = y_train[:n_train_samples]

# Let's now see how the data appears by reshaping the first few samples into 28x28 pixels images:
X_train_img = X_train.reshape(-1, 28, 28)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    ax[i].imshow(X_train_img[i, :, :], cmap='gray')
    ax[i].set_title(f'Label: {y_train[i]}')
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()


# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# Train the classifier
classifier.fit(X_train, y_train)

# Now it's time to check how this classifier performs on the test set. Let's load and rescale it first:

test_data = np.genfromtxt('/Users/tobiadewunmi/Desktop/compVis/l6/sample_data/mnist_test.csv', delimiter=",")
print(test_data.shape)

X_test = test_data[:, 1:]
y_test = test_data[:, 0]

# Rescale the intensities, cast the labels to int
X_test = X_test / 255.
y_test = y_test.astype(int)

# We can now predict the classes on the images of the test set using the predict method:
y_pred = classifier.predict(X_test)

# Let's check the result qualitatively on a small (and randomly shuffled) subset of the test set:
X_test, y_test, y_pred = shuffle(X_test, y_test, y_pred)
X_test_img = X_test.reshape(-1, 28, 28)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    ax[i].imshow(X_test_img[i, :, :], cmap='gray')
    ax[i].set_title(f'Label: {y_test[i]}\n Prediction: {y_pred[i]}')
    ax[i].set_axis_off()
fig.tight_layout
plt.show()

# checking overall performance
# confusion matrix
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
# classigication report
print(f"""Classification report for classifier {classifier}:\n
      {metrics.classification_report(y_test, y_pred)}""")