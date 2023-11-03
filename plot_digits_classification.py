from sklearn import metrics, svm
from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load
import pandas as pd

n_runs = 1

# Data
X, y = read_digits()

# Hyperparameter combinations
classifier_params = {}
gamma_list = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
C_list = [0.1, 1, 10, 100, 1000]
h_params = {'gamma': gamma_list, 'C': C_list}
h_params_combinations = get_hyperparameter_combinations(h_params)
classifier_params['svm'] = h_params_combinations

max_depth_list = [5, 10, 15, 20, 50, 100]
h_params_tree = {'max_depth': max_depth_list}
h_params_trees_combinations = get_hyperparameter_combinations(h_params_tree)
classifier_params['tree'] = h_params_trees_combinations

results = []
test_sizes = [0.2]
dev_sizes = [0.2]

for cur_run_i in range(n_runs):
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1 - test_size - dev_size
            X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)

            binary_preds = {}
            model_preds = {}
            for model_type in classifier_params:
                current_hparams = classifier_params[model_type]
                # breakpoint()
                best_hparams, best_model_path, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, current_hparams, model_type)
                best_model = load(best_model_path)

                test_acc, test_f1, predicted_y = predict_and_eval(best_model, X_test, y_test)
                train_acc, train_f1, _ = predict_and_eval(best_model, X_train, y_train)
                dev_acc = best_accuracy

                # print("{}\ttest_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}, test_f1={:.2f}".format(model_type, test_size, dev_size, train_size, train_acc, dev_acc, test_acc, test_f1))
                cur_run_results = {'model_type': model_type, 'run_index': cur_run_i, 'train_acc': train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}
                results.append(cur_run_results)
                binary_preds[model_type] = y_test == predicted_y
                model_preds[model_type] = predicted_y

#                 print("{}-GroundTruth Confusion metrics".format(model_type))
#                 print(metrics.confusion_matrix(y_test, predicted_y))

# print("svm-tree Confusion metrics".format())
# print(metrics.confusion_matrix(model_preds['svm'], model_preds['tree']))

# print("binarized predictions")
# print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False]))
# print("binarized predictions -- normalized over true labels")
# print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False], normalize='true'))
# print("binarized predictions -- normalized over pred  labels")
# print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False], normalize='pred'))

# result_df = pd.DataFrame(results)
# print(result_df.groupby('model_type').describe().T)