import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

iterations = 500

def f1(x):
    return x


def f2(x):
    return 1-(1/4*math.pi**2)*(x[0]+math.pi)**2+abs(x[1]-5*math.cos(x[0]))**(1/3)+abs(x[2]-5*math.sin(x[0]))**(1/3)


def f(x):
    global w1, w2
    return w1*f1(x[0]) + w2*f2(x)


x = np.ndarray(shape=(iterations, 3))
w = np.ndarray(shape=(iterations, 2))
y = np.ndarray(shape=(iterations, 2))

for i in range(iterations):
    x1 = np.random.uniform(-math.pi, math.pi)
    x2 = np.random.uniform(-5, 5)
    x3 = np.random.uniform(-5, 5)

    w1 = np.random.uniform(0, 1)
    w2 = 1 - w1
    w[i, :] = w1, w2

    args = [x1, x2, x3]
    minimized = minimize(f, args, method='nelder-mead')
    x[i, :] = minimized['x']
    y[i, :] = [f1(x1), f2([x1, x2, x3])]

plt.scatter(y[:, 0], y[:, 1])
plt.ylim(-100, 10)
plt.show()