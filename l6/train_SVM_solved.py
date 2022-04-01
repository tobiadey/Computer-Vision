##############################################################################
# TODO: Implementing a different type of SVM                                 #
##############################################################################

from sklearn import svm

def train_linear_SVM(X_train, y_train):
  """Train linear SVM"""
  classifier = svm.SVC(kernel='linear')
  classifier.fit(X_train, y_train)
  return classifier

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################