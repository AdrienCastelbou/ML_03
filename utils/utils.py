import numpy as np

def data_spliter(x, y, proportion):
    try:
        if type(x) != np.ndarray or type(y) != np.ndarray or type(proportion) != float:
            return None
        n_train = int(proportion * x.shape[0])
        n_test = int((1-proportion) * x.shape[0]) + 1
        perm = np.random.permutation(len(x))
        s_x = x[perm]
        s_y = y[perm]
        x_train, y_train = s_x[:n_train], s_y[:n_train]
        x_test, y_test =  s_x[-n_test:], s_y[-n_test:]
        return x_train, x_test, y_train, y_test
    except:
        return None