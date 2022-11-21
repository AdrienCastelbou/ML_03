import numpy as np
import math

def sigmoid_(x):
    try:
        if type(x) != np.ndarray:
            return None
        if not len(x) or x.shape[1] != 1:
            return None
        sig_x = np.zeros(x.shape)
        for i in range(len(x)):
            sig_x[i] = 1 / (1 + math.e ** -x[i])
        return sig_x
    except:
        return None


def main_test():
    x = np.array([[-4]])
    print(sigmoid_(x))
    x = np.array([[2]])
    print(sigmoid_(x))
    x = np.array([[-4], [2], [0]])
    print(sigmoid_(x))

if __name__ == "__main__":
    main_test()