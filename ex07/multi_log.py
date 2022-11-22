import sys
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils.utils import data_spliter
from ex06.my_logistic_regression import MyLogisticRegression
import matplotlib.pyplot as plt


def engine_Y(Y_train, reference):
    for i in range(Y_train.shape[0]):
        if reference == float(Y_train[i]):
            Y_train[i] = 1.
        else:
            Y_train[i] = 0.
    return Y_train

def load_datasets():
    content = pd.read_csv("solar_system_census.csv")
    X = np.array(content[["weight", "height", "bone_density"]])
    if X.shape[1] !=  3:
        raise Exception("Datas are missing in solar_system_census.csv")        
    content = pd.read_csv("solar_system_census_planets.csv")
    Y = np.array(content[["Origin"]])
    if Y.shape[1] !=  1:
        raise Exception("Datas are missing in solar_system_census_planets.csv")   
    return X, Y

def train_classifier(X_train, Y_train, reference):
    myLR = MyLogisticRegression(theta=np.random.rand(X_train.shape[1] + 1, 1).reshape(-1, 1), max_iter=500000)
    Y_train = engine_Y(np.copy(Y_train), reference)
    myLR.fit_(X_train, Y_train)
    return myLR

def vizualize_preds(X, Y, pred):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(12.5, 5.5)
    fig.tight_layout()
    ax1.scatter(X[:,0], Y, label="Real values")
    ax1.scatter(X[:,0], pred, label="Predictions")
    ax1.grid()
    ax1.legend()
    ax1.set_xlabel("weight")
    ax1.set_ylabel("Origin")
    ax2.scatter(X[:,1], Y, label="Real values")
    ax2.scatter(X[:,1], pred, label="Predictions")
    ax2.grid()
    ax2.legend()
    ax2.set_xlabel("height")
    ax2.set_ylabel("Origin")
    ax3.scatter(X[:,2], Y, label="Real values")
    ax3.scatter(X[:,2], pred, label="Predictions")
    ax3.grid()
    ax3.legend()
    ax3.set_xlabel("bone_density")
    ax3.set_ylabel("Origin")
    fig.suptitle("Predictions comparisions")
    plt.show()

def perform_multi_classification(X, Y):
    X_train, X_test, Y_train, Y_test = data_spliter(X, Y, 0.8)
    classifiers = []
    for i in range(4):
        classifiers.append(train_classifier(X_train, Y_train, i))
    preds = []
    for classifier in classifiers:
        preds.append(classifier.predict_(X_test))
    y_hat = np.zeros(Y_test.shape)
    for i, cl_zero_pred, cl_one_pred, cl_two_pred, cl_three_pred in zip(range(y_hat.shape[0]), preds[0], preds[1], preds[2], preds[3]):
        best = max(cl_zero_pred, cl_one_pred, cl_two_pred, cl_three_pred)
        if best == cl_zero_pred:
            y_hat[i] = 0
        elif best == cl_one_pred:
            y_hat[i] = 1
        elif best == cl_two_pred:
            y_hat[i] = 2
        elif best == cl_three_pred:
            y_hat[i] = 3
    print(f'Precision : {len(y_hat[y_hat == Y_test])} / {len(y_hat)}')
    vizualize_preds(X_test, Y_test, y_hat)
    
        

def main():
    try:
        X, Y = load_datasets()
    except Exception as e:
        print("Error in datas loading :", e)
        return
    perform_multi_classification(X, Y)   
    

if __name__ == "__main__":
    main()