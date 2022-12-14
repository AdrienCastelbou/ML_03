import numpy as np
import sys
sys.path.append('../')
from ex01.log_pred import logistic_predict_

def vec_log_loss_(y, y_hat, eps=1e-15):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or type(eps) != float:
            return None
        if y.shape[1] != 1 or y.shape != y_hat.shape:
            return None
        l = len(y)
        v_ones = np.ones((l, 1))
        return - float(y.T.dot(np.log(y_hat + eps)) + (v_ones - y).T.dot(np.log(v_ones - y_hat + eps))) / l
    except:
        return None


def main_test():
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    print(vec_log_loss_(y1, y_hat1))
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    print(vec_log_loss_(y2, y_hat2))
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    print(vec_log_loss_(y3, y_hat3))


if __name__ == "__main__":
    main_test()