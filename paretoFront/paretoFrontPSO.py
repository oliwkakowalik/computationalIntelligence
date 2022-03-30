import numpy as np
import matplotlib.pyplot as plt
import math

iterations = 7000
birdsNumber = 512


def f1(x: int):
    if any([x < -math.pi, x > math.pi]):
        return abs(x) * 1000000
    else:
        return x


def f2(x):
    temp = 1 - (1 / (4 * (math.pi ** 2))) * ((x[0] + math.pi) ** 2) + (abs(x[1] - 5 * math.cos(x[0]))) ** (1 / 3) + (
        abs(x[2] - 5 * math.sin(x[0]))) ** (1 / 3)
    if any([x[0] < -math.pi, x[0] > math.pi, x[1] < -5, x[1] > 5, x[2] < -5, x[2] < -5]):
        return abs(temp) * 1000000
    else:
        return temp


def f(x):
    return f1(x[0]) + f2(x)


###pso bare bones###
#inicjalizacja
birds = np.ndarray((birdsNumber, 5))
birds[:, 0] = np.random.uniform(-math.pi, math.pi, birdsNumber)
birds[:, 1] = np.random.uniform(-5, 5, birdsNumber)
birds[:, 2] = np.random.uniform(-5, 5, birdsNumber)
birds[:, 3] = [f1(x) for x in birds[:, 0]]
birds[:, 4] = [f2(x) for x in birds[:, 0:3]]


# ustalamy które rozwiązanie jest najlepsze
bestBird1 = birds[np.argmin(birds[:, 3]), :]
bestBird2 = birds[np.argmin(birds[:, 4]), :]

for i in range(iterations):
    for j in range(birdsNumber):
        # odchylenia standardowe dla wymiaru x i y
        sigmaX1 = max(abs(birds[j, 0] - bestBird1[0]), abs(birds[j, 0] - bestBird2[0]))
        sigmaX2 = max(abs(birds[j, 1] - bestBird1[1]), abs(birds[j, 1] - bestBird2[1]))
        sigmaX3 = max(abs(birds[j, 2] - bestBird1[2]), abs(birds[j, 2] - bestBird2[2]))

        # testowe pozycje i wartość funkcji
        testX1 = np.random.normal((birds[j, 0] + bestBird1[0] + bestBird2[0]) / 3, sigmaX1, 1)
        testX2 = np.random.normal((birds[j, 1] + bestBird1[1] + bestBird2[1]) / 2, sigmaX2, 1)
        testX3 = np.random.normal((birds[j, 2] + bestBird1[2] + bestBird2[2]) / 3, sigmaX3, 1)
        testF1 = f1(testX1)
        testF2 = f2([testX1, testX2, testX3])

        # jeśli jestem lepszy niż poprzedni robię hop
        if all([testF1 <= birds[j, 3], testF2 <= birds[j, 4]]):
            birds[j, 0] = testX1
            birds[j, 1] = testX2
            birds[j, 2] = testX3
            birds[j, 3] = testF1
            birds[j, 4] = testF2

    #który ptaszek jest teraz najlepszy?
    bestBird1 = birds[np.argmin(birds[:, 3]), :]
    bestBird2 = birds[np.argmin(birds[:, 4]), :]

# dorysowanie ścieżki znajdowania rozwiązania
plt.scatter(birds[:, 3], birds[:, 4])
plt.show()

# rozwiązanie najbliższe utopijnemu
f1s = (birds[:, 3] - min(birds[:, 3]))/(max(birds[:, 3]) - min(birds[:, 3]))
f2s = (birds[:, 4] - min(birds[:, 4]))/(max(birds[:, 4]) - min(birds[:, 4]))
dist = [math.sqrt(f1s[i]**2+f2s[i]**2) for i in range(birdsNumber)]
bestBird = birds[np.argmin(dist), :]
print(bestBird)