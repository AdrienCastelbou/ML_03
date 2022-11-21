import sys
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils.utils import data_spliter
from ex06.my_logistic_regression import MyLogisticRegression
import matplotlib.pyplot as plt


def get_arg():
    if len(sys.argv) != 2:
        raise Exception("wrong number of arguments")
    s = sys.argv[1]
    param = s.split("=")
    if param[0] != "-zipcode":
        raise Exception("wrong arg name")
    zipcode = int(param[1])
    if (zipcode < 0 or zipcode > 3):
        raise Exception("wrong arg value")
    return zipcode

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

def engine_Y(Y_train, reference):
    for i in range(Y_train.shape[0]):
        if reference == float(Y_train[i]):
            Y_train[i] = 1.
        else:
            Y_train[i] = 0.
    return Y_train

def vizualize_preds(X, Y, pred):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Predictions comparisions")
    ax1.scatter(X[:,0], Y, label="Real values")
    ax1.scatter(X[:,0], pred, label="Predictions")
    ax1.legend()
    ax1.set_xlabel("weight")
    ax1.set_ylabel("is_from_planet")
    ax2.scatter(X[:,1], Y, label="Real values")
    ax2.scatter(X[:,1], pred, label="Predictions")
    ax2.legend()
    ax2.set_xlabel("height")
    ax2.set_ylabel("is_from_planet")
    ax3.scatter(X[:,2], Y, label="Real values")
    ax3.scatter(X[:,2], pred, label="Predictions")
    ax3.legend()
    ax3.set_xlabel("bone_density")
    ax3.set_ylabel("is_from_planet")
    plt.show()

def threshold_datas(y, threshold=0.5):
    for i in range(len(y)):
        if y[i] >= threshold:
            print(y[i])
            y[i] = 1
        else:
            y[i] = 0
    print(len(y[y == 1]))
    return y
    y[ y >= threshold] = 1
    y[y < threshold] = 0
    return y

def perform_classification(X, Y, zipcode):
    X_train, X_test, Y_train, Y_test = data_spliter(X, Y, 0.8)
    myLR = MyLogisticRegression(theta=np.random.rand(X_train.shape[1] + 1, 1).reshape(-1, 1), max_iter=1000)
    Y_train = engine_Y(Y_train, zipcode)
    Y_HAT = myLR.predict_(X_train)
    print(myLR.loss_(y=Y_train, y_hat=Y_HAT))
    myLR.fit_(X_train, Y_train)
    Y_HAT = myLR.predict_(X_train)
    print(myLR.loss_(y=Y_train, y_hat=Y_HAT))
    Y_HAT = myLR.predict_(X_test)
    Y_HAT = threshold_datas(Y_HAT)
    Y_test = engine_Y(Y_test, zipcode)
    print(len(Y_test[Y_test == 1]), len(Y_test[Y_test == 0]), len(Y_HAT[Y_HAT == 1]), len(Y_HAT[Y_HAT == 0]))
    #vizualize_preds(X_test, Y_test, Y_HAT)

def main():
    try:
        zipcode = get_arg()
    except Exception as e:
        print(f"{e}, use -zipcode=X with X being 0, 1, 2 or 3 to start")
        return
    try:
        X, Y = load_datasets()
    except Exception as e:
            print("Error in datas loading :", e)
            return
    perform_classification(X, Y, zipcode)   
    

if __name__ == "__main__":
    main()