from sklearn import metrics
from sklearn.model_selection import train_test_split

def split_train_dev_test(X, y, test_size, dev_size):

    # Split the data into train data and rest of the data
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=(test_size + dev_size), random_state=42)

    # Calculate the dev_ratio
    dev_ratio = dev_size / (test_size + dev_size)

    # Split the rest of the data into dev data and test data
    X_dev, X_test, y_dev, y_test = train_test_split(X_rest, y_rest, test_size=dev_ratio, random_state=42)

    return X_train, y_train, X_dev, y_dev, X_test, y_test

def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)

    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    
    return disp