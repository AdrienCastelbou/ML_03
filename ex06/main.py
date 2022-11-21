import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR
X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
Y = np.array([[1], [0], [1]])
thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
mylr = MyLR(thetas)
# Example 0:
Y_HAT = mylr.predict_(X)
print(Y_HAT)
print(mylr.loss_(Y,Y_HAT))
print(mylr.theta)
mylr.fit_(X, Y)
Y_HAT = mylr.predict_(X)
print(Y_HAT)
print(mylr.loss_(Y, Y_HAT))
print(mylr.loss_elem_(Y, Y_HAT))