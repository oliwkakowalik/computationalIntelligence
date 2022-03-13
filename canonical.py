#2) Przerobić istniejący skrypt tak aby zamiast wariantu “bare bones” realizował
#obliczenia w oparciu o wariant kanoniczny (patrz wykład).

#Uniform Distribution- rozkład równomierny np.random.uniform

import numpy as np
import matplotlib.pyplot as plt

#funkcja testowa
def rosen(x):
    return 100 * (x[1] - x[0] * x[0])**2 + (1 - x[0])**2

#narysujmy ją w zakresie od -2 do 2
x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)

n = len(x)

X, Y = np.meshgrid(x, y)
Z = rosen((X, Y))

#inicjalizacja
bx = np.random.uniform(-2, 2, 16)
by = np.random.uniform(-2, 2, 16)

#budujemy ramkę z pozycjami i wartością funkcji celu
z = np.array([rosen((bx[i], by[i])) for i in range(0, 16)])
b = np.array([(bx[i], by[i], z[i], 0, 0, bx[i], by[i], z[i]) for i in range(0, 16)])
#tablica postaci:
#bieżący wektor położenia
#bieżący wektor prędkości (tylko x y)
#najlepszy wektor położenia

#ustalamy położenie najlepszego rozwiązania
bestBird = b[np.argmin(b[:, 2]), 0:3]

#budujemy pustą ramkę na zapisanie evolucji rozwiązania
bestEvo = np.zeros((300, 3))

#parameters
suppression = 0.7298 #tłumienie
phiIndv = 2.05
phiSoc = 2.05

for i in range(300):
    for j in range(16):
        uIndv = np.random.uniform(0, phiIndv, 1)
        uSoc = np.random.uniform(0, phiSoc, 1)
        diffIndv = b[j, 5:7] - b[j, 0:2]
        diffSoc = bestBird[0:2] - b[j, 0:2]

        b[j, 3:5] = suppression*(b[j, 3:5]+uIndv*diffIndv + uSoc*diffSoc)
        b[j, 0:2] = b[j, 0:2] + b[j, 3:5]
        b[j, 2] = rosen(b[j, 0:2])

        if b[j, 2] < b[j, 7]:
            b[j, 5:] = b[j, 0:3]

    #który ptaszek jest teraz najlepszy?
    #wybieram najlepszego ptaska z iteracjim tak ma byc?
    #czy mam wybrac najlepszegp ptaszka ze wszytskich poprzednich
    bestBird = b[np.argmin(b[:, 2]), 0:3]

    #zapis ewolucji rozwiązania
    bestEvo[i] = bestBird

#dorysowanie ścieżki znajdowania rozwiązania
plt.contour(X, Y, Z, levels=[x**2.5 for x in range(2, 25)], cmap="plasma")
plt.scatter(bestEvo[:, 0], bestEvo[:, 1], marker='o', color='black', alpha=0.2, linewidths=0.1)
plt.show()

print(bestEvo)