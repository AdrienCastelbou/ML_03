import numpy as np
import pandas as pd


def confusion_matrix(y_true, y_hat, labels=None, df_option=False):
    try:
        if type(y_true) != np.ndarray or type(y_hat) != np.ndarray:
            return None
        if (labels != None and type(labels) != list) or type(df_option) != bool:
            return None
        if labels == None:
            labels = np.unique((y_true, y_hat)).tolist()
        c_matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for y_true_i, y_hat_i in zip(y_true, y_hat):
            if y_true_i.item() in labels and y_hat_i.item() in labels:
                c_matrix[labels.index(y_true_i.item())][labels.index(y_hat_i.item())] += 1
        if df_option == True:
            return pd.DataFrame(c_matrix, index=labels, columns=labels)
        return c_matrix
    except:
        return None

def main_test():
    y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])
    print(confusion_matrix(y, y_hat))
    print(confusion_matrix(y, y_hat, ["dog", "norminet"]))
    print(confusion_matrix(y, y_hat, df_option=True))
    print(confusion_matrix(y, y_hat, ["bird", "dog"], df_option=True))


if __name__ == "__main__":
    main_test()