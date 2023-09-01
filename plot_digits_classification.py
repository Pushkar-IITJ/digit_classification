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

# Predict the value of the digit on the test subset
disp = predict_and_eval(clf, X_test, y_test)

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)