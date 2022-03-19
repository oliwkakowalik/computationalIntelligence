#cannocial PSO, working for n dimensions, with stop condition and possibility of printing result as plot for 2D
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt

# parameters
birdsNumber = 16
dims = 2

phiIndv = 2.05
phiSoc = 2.05
phiSum = phiIndv + phiSoc
suppression = 2 / (phiSum - 2 + math.sqrt(phiSum ** 2 - 4 * phiSum))

wantedPoint = np.array([1, 1])

def rosenbrock(x):
    return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


def initSwarm(birdsNum: int, dimensions: int):
    currPos = np.array([np.random.uniform(-2, 2, dimensions) for i in range(birdsNum)])  # (birdsNumber, dimensions)
    bestPos = np.copy(currPos)  # (birdsNumber, dimensions)
    v = np.array([np.random.uniform(-1, 1, dimensions) for i in range(birdsNum)])  # (birdsNumber, dimensions)
    fValues = np.apply_along_axis(rosenbrock, 1, currPos)  # (birdsNumber, 1)
    bestBird = currPos[np.argmin(fValues), :]  # (1, dimensions)

    return currPos, bestPos, v, fValues, bestBird


def stopCondition(currPos, point: np.array, dimensions):
    avgs = np.array([statistics.mean(currPos[:, i]) for i in range(dimensions)])
    diff = abs(avgs - point)
    return True if all(diff < 0.00000001) else False


def updateBird(currPos, bestPos, v, fValues, index: int):
    global phiIndv, phiSoc, suppression, bestBird

    uIndv = np.random.uniform(0, phiIndv, 1)
    uSoc = np.random.uniform(0, phiSoc, 1)
    diffIndv = bestPos[index, :] - currPos[index, :]
    diffSoc = bestBird - currPos[index, :]

    v[index, :] = suppression * (v[index, :] + uIndv * diffIndv + uSoc * diffSoc)
    currPos[index, :] += v[index, :]
    fValues[index] = rosenbrock(currPos[index, :])
    bestPos[index, :] = bestPos[index, :] if rosenbrock(bestPos[index, :]) < fValues[index] else currPos[index, :]


def updateSworm(currPos, bestPos, v, fValues, dimensions, birdsNumber):
    global bestBird, bestBirdEvolution, wantedPoint
    n = 0
    while not stopCondition(currPos, wantedPoint, dimensions):
        for i in range(birdsNumber):
            updateBird(currPos, bestPos, v, fValues, i)
        bestBird = currPos[np.argmin(fValues), :]
        bestBirdEvolution[n, :] = bestBird
        n += 1
    return n-1, bestBird

def printEvolution2D(evolution, n: int):
    x = np.arange(-2, 2, 0.1)

    xy = np.array([[xi, yi] for xi in x for yi in x])
    X, Y = np.meshgrid(x, x)
    Z = np.apply_along_axis(rosenbrock, 1, xy)
    Z = np.reshape(Z, newshape=(40, 40))

    plt.contour(X, Y, Z, levels=[x**2.5 for x in range(2, 25)], cmap="plasma")
    plt.scatter(evolution[:n, 0], evolution[:n, 1], marker='o', color='black', alpha=0.2, linewidths=0.1)
    plt.show()


bestBirdEvolution = np.zeros(shape=(10000, dims))

currentPosition, bestPosition, velocity, functionValues, bestBird = initSwarm(birdsNumber, dims)
numberOfIterations, point = updateSworm(currentPosition, bestPosition, velocity, functionValues, dims, birdsNumber)

print("point found:", point, "in", numberOfIterations, "th iteration")
if dims == 2:
    printEvolution2D(np.array(bestBirdEvolution), numberOfIterations)