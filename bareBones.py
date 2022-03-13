import numpy as np
import matplotlib.pyplot as plt
import pandas

#funkcja testowa
def rosen(x):
    return 100 * (x[1] - x[0] * x[0])**2 + (1 - x[0])**2

#narysujmy ją w zakresie od -2 do 2
x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)

n = len(x)

X, Y = np.meshgrid(x, y)
Z = rosen((X, Y))

###pso bare bones###
#inicjalizacja
bx = np.random.uniform(-2, 2, 16)
by = np.random.uniform(-2, 2, 16)

#dataframe z pozycjami i wartością funkcji celu
z = np.array([rosen((bx[i], by[i])) for i in range(0, 16)])
b = np.array([(bx[i], by[i], z[i]) for i in range(0, 16)])

#ustalamy które rozwiązanie jest najlepsze
bestBird = b[np.argmin(b[:, 2]), :]

#budujemy pustą ramkę na zapisanie evolucji rozwiązania
bestEvo = np.zeros((100, 3))

for i in range(100):
    for j in range(16):
        #odchylenia standardowe dla wymiaru x i y
        sigmaX = abs(b[j, 0] - bestBird[0])
        sigmaY = abs(b[j, 1] - bestBird[1])
        #testowe pozycje i wartość funkcji
        testX = np.random.normal((b[j, 0] + bestBird[0])/2, sigmaX, 1)
        testY = np.random.normal((b[j, 1] + bestBird[1])/2, sigmaY, 1)
        testZ = rosen((testX, testY))
        #jeśli jestem lepszy niż poprzedni robię hop
        if testZ < b[j, 2]:
            b[j, 0] = testX
            b[j, 1] = testY
            b[j, 2] = testZ

    # #który ptaszek jest teraz najlepszy?
    bestBird = b[np.argmin(b[:, 2]), :]

    #zapis ewolucji rozwiązania
    bestEvo[i] = bestBird

#dorysowanie ścieżki znajdowania rozwiązania
plt.contour(X, Y, Z, levels=[x**2.5 for x in range(2, 25)], cmap="plasma")
plt.scatter(bestEvo[:, 0], bestEvo[:, 1], marker='o', color='black', alpha=0.2, linewidths=0.1)
plt.show()
