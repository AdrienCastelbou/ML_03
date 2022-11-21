import numpy as np
import sys
sys.path.append('../')
from ex01.log_pred import logistic_predict_

def log_gradient_(x, y, theta):
    try:
        if type(x) != np.ndarray or type(y) != np.ndarray or type(theta) != np.ndarray:
            return None
        if theta.shape[0] != x.shape[1] + 1:
            return None
        if y.shape[1] != 1:
            return None
        l = len(y)
        nabla_J = np.zeros(theta.shape)
        for x_i, y_i, y_hat_i in zip(x, y, logistic_predict_(x, theta)):
            for j in range(theta.shape[0]):
                if j == 0:
                    nabla_J[j] += (y_hat_i - y_i)
                else:
                    nabla_J[j] += (y_hat_i - y_i) * x_i[j - 1]
        return nabla_J / l
    except:
        return None


def main_test():
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    print(log_gradient_(x1, y1, theta1))
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(log_gradient_(x2, y2, theta2))
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(log_gradient_(x3, y3, theta3))

if __name__ == "__main__":
    main_test()