"""
================================
Recognizing hand-written digits
================================

This code shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""
import numpy as np
from sklearn import  metrics, svm
from sklearn.tree import DecisionTreeClassifier
from utils import split_train_dev_test, predict_and_eval, read_data, model_training, param_combination, tune_hparams, test_dev_variations

X, y = read_data()

# Split data into 70% train, 15% dev and 15% test subsets
X_train, y_train, X_dev, y_dev, X_test, y_test = split_train_dev_test(X, y, test_size=0.15, dev_size=0.15)

# # Create a classifier: a support vector classifier
# clf = model_training(X_train,y_train,{'gamma':0.001},model='svm')

# # Predict and evaluate the results
# predict_and_eval(clf,X_test,y_test)

# assignment 3: Task 2
# Hyper parameter all possible combinations
list_of_all_param_combination=param_combination(model='svm')
best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination)
production_model = model_training(X_train, y_train, best_hparams, model='svm')
production_accuracy = predict_and_eval(production_model, X_test, y_test)

candidate_hparams = {'max_depth': 10, 'min_samples_split': 2}
candidate_model = DecisionTreeClassifier(**candidate_hparams)
candidate_model.fit(X_train, y_train)
candidate_accuracy = predict_and_eval(candidate_model, X_test, y_test)


production_predictions = production_model.predict(X_test)
candidate_predictions = candidate_model.predict(X_test)

# confusion_matrix = metrics.confusion_matrix(y_test, production_predictions)
confusion_matrix = metrics.confusion_matrix(production_predictions, candidate_predictions)

true_positives_production = np.logical_and(production_predictions == y_test, candidate_predictions == production_predictions).sum()

false_negatives_production = np.logical_and(production_predictions == y_test, candidate_predictions != production_predictions).sum()

false_positives_production = np.logical_and(candidate_predictions == y_test, candidate_predictions != production_predictions).sum()

true_negatives_production = np.logical_and(candidate_predictions != y_test, candidate_predictions == production_predictions).sum()


confusion_matrix_2x2 = np.array([[true_positives_production, false_negatives_production],
                                  [false_positives_production, true_negatives_production]])

# Production Macro-average F1 score
production_macro_avg_f1 = metrics.f1_score(y_test, production_predictions, average='macro')

# Candidate Macro-average F1 score
candidate_macro_avg_f1 = metrics.f1_score(y_test, candidate_predictions, average='macro')

print(f"Production model's accuracy: {production_accuracy}")
print(f"Candidate model's accuracy: {candidate_accuracy}")
print("Confusion Matrix:")
print(confusion_matrix)
print("confusion_matrix_2x2:")
print(confusion_matrix_2x2)
print(f"Macro-average F1 score: {production_macro_avg_f1}")
print(f"Macro-average F1 score: {candidate_macro_avg_f1}")