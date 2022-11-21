import numpy as np
import math

def logistic_predict_(x, theta):
    try:
        if type(x) != np.ndarray or type(theta) != np.ndarray:
            return None
        if not len(x) or not len(theta):
            return None
        extended_x = np.hstack((np.ones((x.shape[0], 1)), x))
        return 1 / (1 + math.e ** - extended_x.dot(theta))
    except:
        return None

def main_test():
    x = np.array([4]).reshape((-1, 1))
    theta = np.array([[2], [0.5]])
    print(logistic_predict_(x, theta))
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(logistic_predict_(x2, theta2))
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(logistic_predict_(x3, theta3))

if __name__ == "__main__":
    main_test()