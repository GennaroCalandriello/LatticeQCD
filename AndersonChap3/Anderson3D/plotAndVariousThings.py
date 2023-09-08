import numpy as np
import matplotlib.pyplot as plt

from module.functions import *


def plotDistribution():
    s = np.linspace(0, 3, 1000)

    plt.figure()
    plt.plot(distribution(s, "GOE"), "g--", label="GOE")
    plt.plot(distribution(s, "GUE"), "b--", label="GUE")
    plt.plot(distribution(s, "GSE"), "r--", label="GSE")
    plt.plot(distribution(s, "Poisson"), "c--", label="Poisson")
    plt.xlabel("s")
    plt.ylabel(r"$p^\beta_{WD}(s)$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plotDistribution()
