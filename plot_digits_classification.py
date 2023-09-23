"""
================================
Recognizing hand-written digits
================================

This code shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

from sklearn import  metrics, svm
from utils import split_train_dev_test, predict_and_eval, read_data, model_training, param_combination, tune_hparams, test_dev_variations

X, y = read_data()

# Split data into 70% train, 15% dev and 15% test subsets
# X_train, y_train, X_dev, y_dev, X_test, y_test = split_train_dev_test(X, y, test_size=0.15, dev_size=0.15)

# # Create a classifier: a support vector classifier
# clf = model_training(X_train,y_train,{'gamma':0.001},model='svm')

# # Predict and evaluate the results
# predict_and_eval(clf,X_test,y_test)

# assignment 3: Task 2
# Hyper parameter all possible combinations
# list_of_all_param_combination=param_combination(model='svm')

# Best parameters, model and accuracy after Hyper parameter tuning
# best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination)

# assignment 3: Task 3 and 4 
# test_dev_variations(X,y)

# print
print(f"The number of total samples in the dataset: {len(y)}")
print(f"Size (height and width) of the images in dataset: {len(y),len(X[0])}")