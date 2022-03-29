import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

iterations = 2000


def f1(x):
    if x < -math.pi or x > math.pi:
        return abs(x)*1000000
    else:
        return x


def f2(x):
    temp = 1-(1/(4*(math.pi**2)))*((x[0]+math.pi)**2)+(abs(x[1]-5*math.cos(x[0])))**(1/3)+(abs(x[2]-5*math.sin(x[0])))**(1/3)
    if x[0] < -math.pi or x[0] > math.pi or x[1] < -5 or x[1] > 5 or x[2] < -5 or x[2] < -5:
        return abs(temp) * 1000000
    else:
        return temp


def f(x):
    global w1, w2
    return w1*f1(x[0]) + w2*f2(x)


x = np.ndarray(shape=(iterations, 3))
w = np.ndarray(shape=(iterations, 2))
y = np.ndarray(shape=(iterations, 2))
y2 = np.ndarray(shape=(iterations, 2))


for i in range(iterations):
    x1 = np.random.uniform(-math.pi, math.pi)
    x2 = np.random.uniform(-5, 5)
    x3 = np.random.uniform(-5, 5)

    w1 = np.random.uniform(0, 1)
    w2 = 1 - w1
    w[i, :] = w1, w2

    args = np.ndarray((3, 1), dtype=np.float64)
    args = x1, x2, x3
    minimized = minimize(f, args, method='nelder-mead')
    x[i, :] = minimized['x']
    y[i, :] = [f1(x[i, 0]), f2(x[i, :])]
    

plt.scatter(y[:, 0], y[:, 1])
plt.show()
