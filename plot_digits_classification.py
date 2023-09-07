"""
================================
Recognizing hand-written digits
================================

This code shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

from sklearn import datasets, metrics, svm
from utils import split_train_dev_test, predict_and_eval

# Read data
digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 70% train, 15% dev and 15% test subsets
X_train, y_train, X_dev, y_dev, X_test, y_test = split_train_dev_test(data, digits.target, test_size=0.15, dev_size=0.15)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

predict_and_eval(clf,X_test,y_test)