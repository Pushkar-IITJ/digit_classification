from itertools import product
from sklearn import metrics, datasets, svm
from sklearn.model_selection import train_test_split


def read_data():
    # Read data
    digits = datasets.load_digits()

    # flatten the images
    n_samples = len(digits.images)

    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    return X, y


def split_train_dev_test(X, y, test_size, dev_size):

    # Split the data into train data and rest of the data
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=(test_size + dev_size), random_state=42)

    # Calculate the dev_ratio
    dev_ratio = dev_size / (test_size + dev_size)

    # Split the rest of the data into dev data and test data
    X_dev, X_test, y_dev, y_test = train_test_split(X_rest, y_rest, test_size=dev_ratio, random_state=42)

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def model_training(X, y, parameters, model = "svm"):

    if model == "svm":
        clf = svm.SVC
    clf_model = clf(**parameters)

    # Learn the digits on the train subset
    clf_model.fit(X, y)
    return clf_model


def predict_and_eval(model, X_test, y_test):
    
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test,predicted)


def param_combination(model='svm'):

    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]

    if model == 'svm':
        list_of_all_param_combination = list(product(gamma_ranges, C_ranges))

        return list_of_all_param_combination
    

def tune_hparams(X_train, Y_train, x_dev, y_dev, list_of_all_param_combination):

    best_accuracy = -1
    best_model = None
    for i in list_of_all_param_combination:
        current_model = model_training(X_train,Y_train,{'gamma':i[0], 'C':i[1]},model='svm')
        current_accuracy = predict_and_eval(current_model,x_dev,y_dev)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_model = current_model
            best_hparams = {'gamma':i[0], 'C':i[1]}

    return best_hparams, best_model, best_accuracy


def test_dev_variations(X,y):
    test_size = [0.1, 0.2, 0.3]
    dev_size = [0.1, 0.2, 0.3]
    test_dev_data_combinations = list(product(test_size, dev_size))

    for i in test_dev_data_combinations:
        X_train, y_train, X_dev, y_dev, X_test, y_test = split_train_dev_test(X,y,i[0],i[1])
        list_of_all_param_combination=param_combination(model='svm')
        best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination)
        train_acc = predict_and_eval(best_model,X_train,y_train)
        dev_acc = predict_and_eval(best_model,X_dev,y_dev)
        test_acc = predict_and_eval(best_model,X_test,y_test)
        print(f"""test_size={i[0]}, dev_size={i[1]}, train_size={1-sum(i)}, train_acc={train_acc}, 
              dev_acc={dev_acc}, test_acc={test_acc}, best_hparams={best_hparams}""")


