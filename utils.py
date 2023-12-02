from sklearn.model_selection import train_test_split
from sklearn import svm, tree, datasets, metrics
from joblib import dump, load


# Q1 solution:
from sklearn.preprocessing import Normalizer
def preprocess_unit_normalization(X_train, X_test):
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    normalizer = Normalizer()
    X_train_normalized = normalizer.fit_transform(X_train)
    X_test_normalized = normalizer.transform(X_test)

    return X_train_normalized, X_test_normalized

# Q2 solution:
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate_lr(X_train_normalized, y_train, X_test_normalized, y_test, roll_no="M22AIE213"):
    solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    models_dir = 'models'
    save_models = os.environ.get('SAVE_MODELS', 'TRUE')

    for solver in solvers:
        model = LogisticRegression(solver=solver)
        model.fit(X_train_normalized, y_train)

        # Cross-validation for evaluation
        scores = cross_val_score(model, X_train_normalized, y_train, cv=5)
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Mean and Standard Deviation
        print(f"Solver: {solver}, CV Mean Accuracy: {mean_score:.2f}, CV Standard Deviation: {std_score:.2f}")

        # Evaluating on test data
        y_pred = model.predict(X_test_normalized)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Solver: {solver}, Test Accuracy: {test_accuracy:.2f}")
        print(f"Classification Report for Solver {solver}:\n{classification_report(y_test, y_pred)}")

        if save_models != 'FALSE':
            model_filename = os.path.join(models_dir, f"{roll_no}_lr_{solver}.joblib")
            dump(model, model_filename)

def get_combinations(param_name, param_values, base_combinations):
    new_combinations = []
    for value in param_values:
        for combination in base_combinations:
            combination[param_name] = value
            new_combinations.append(combination.copy())
    return new_combinations

def get_hyperparameter_combinations(dict_of_param_lists):
    base_combinations = [{}]
    for param_name, param_values in dict_of_param_lists.items():
        base_combinations = get_combinations(param_name, param_values, base_combinations)
    return base_combinations

def tune_hparams(X_train, y_train, X_dev, y_dev, h_params_combinations, model_type="svm"):
    best_accuracy = -1
    best_model_path = ""
    for h_params in h_params_combinations:
        model = train_model(X_train, y_train, h_params, model_type=model_type)
        cur_accuracy, _, _ = predict_and_eval(model, X_dev, y_dev)
        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            best_hparams = h_params
            best_model_path = f"./models/{model_type}_" + "_".join([f"{k}:{v}" for k, v in h_params.items()]) + ".joblib"
            best_model = model
        dump(best_model, best_model_path)
    return best_hparams, best_model_path, best_accuracy

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y

def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
    return X_train, X_test, y_train, y_test

def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        clf = svm.SVC
    if model_type == "tree":
        clf = tree.DecisionTreeClassifier
    model = clf(**model_params)
    model.fit(x, y)
    return model

def train_test_dev_split(X, y, test_size, dev_size):
    X_train_dev, X_test, Y_train_Dev, y_test = split_data(X, y, test_size=test_size, random_state=1)
    print("train+dev = {} test = {}".format(len(Y_train_Dev), len(y_test)))
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, Y_train_Dev, dev_size / (1 - test_size), random_state=1)
    return X_train, X_test, X_dev, y_train, y_test, y_dev

def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test, predicted), metrics.f1_score(y_test, predicted, average="macro"), predicted
