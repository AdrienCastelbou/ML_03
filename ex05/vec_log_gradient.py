import numpy as np
import sys
sys.path.append('../')
from ex01.log_pred import logistic_predict_

def vec_log_gradient(x, y, theta):
        try:
            if type(x) != np.ndarray or type(y) != np.ndarray or type(theta) != np.ndarray:
                return None
            l = len(x)
            y_hat = logistic_predict_(x, theta)
            x = np.hstack((np.ones((x.shape[0], 1)), x))
            nabla_J = x.T.dot(y_hat - y) / l
            return nabla_J
        except:
            return None

def main_test():
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    print(vec_log_gradient(x1, y1, theta1))
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(vec_log_gradient(x2, y2, theta2))
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(vec_log_gradient(x3, y3, theta3))

if __name__ == "__main__":
    main_test()