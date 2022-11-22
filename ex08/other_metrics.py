import numpy as np

def accuracy_score_(y, y_hat):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or y.shape != y_hat.shape:
            return None
        t = 0
        for y_i, y_hat_i in zip(y, y_hat):
            if y_i == y_hat_i:
                t += 1
        return t / len(y)
    except:
        return None

def precision_score_(y, y_hat, pos_label=1):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or y.shape != y_hat.shape or not isinstance(pos_label, (str, int)):
            return None
        tp, fp = 0, 0
        for y_i, y_hat_i in zip(y, y_hat):
            if y_hat_i == pos_label and y_i == y_hat_i:
                tp += 1
            elif y_hat_i == pos_label and y_i != y_hat_i:
                fp += 1
        return tp / (tp + fp)
    except:
        return None

def recall_score_(y, y_hat, pos_label=1):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or y.shape != y_hat.shape or not isinstance(pos_label, (str, int)):
            return None
        tp, fn = 0, 0
        for y_i, y_hat_i in zip(y, y_hat):
            if y_hat_i == pos_label and y_i == y_hat_i:
                tp += 1
            elif y_i == pos_label and y_i != y_hat_i:
                fn += 1
        return tp / (tp + fn)
    except:
        return None

def f1_score_(y, y_hat, pos_label=1):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or y.shape != y_hat.shape or not isinstance(pos_label, (str, int)):
            return None
        prec = precision_score_(y, y_hat, pos_label)
        recall = recall_score_(y, y_hat, pos_label)
        return (2 * prec * recall) / (prec + recall)
    except:
        return None


def evaluate(y, y_hat, label=1):
    print(accuracy_score_(y, y_hat))
    print(precision_score_(y, y_hat, label))
    print(recall_score_(y, y_hat, label))
    print(f1_score_(y, y_hat, label))

def main_test():
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
    evaluate(y, y_hat)
    print('---')
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    evaluate(y, y_hat, 'dog')
    print('---')
    evaluate(y, y_hat, 'norminet')


if __name__ == "__main__":
    main_test()