import numpy as np
import matplotlib.pyplot as plt

from module.functions import *


def plotDistribution():
    s = np.linspace(0, 4, 1000)

    plt.figure(figsize=(8, 7))
    plt.plot(s, distribution(s, "GOE"), "g--", label="GOE")
    plt.plot(s, distribution(s, "GUE"), "b--", label="GUE")
    plt.plot(s, distribution(s, "GSE"), "r--", label="GSE")
    plt.plot(s, distribution(s, "Poisson"), "c--", label="Poisson")
    plt.xlabel("s", fontsize=16)
    plt.ylabel(r"$p(s)$", fontsize=16)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plotGOE():
    w4 = np.loadtxt("results/GOE/eigenvalues/eigenvalues_W_4.0.txt")
    w4 = np.loadtxt("results/GOE/eigenvalues/eigenvalues_W_15.0.txt")

    unfolded = unfold_spectrum(w4, kind="diff")[1]
    # print(unfoldw4)
    p = distribution(unfolded, "GOE")
    plt.title("ULSD for unitary Anderson Hamiltonian", fontsize=16)
    plt.xlabel("s", fontsize=12)
    plt.ylabel(r"p(s)", fontsize=12)
    plt.legend()
    plt.hist(
        unfolded,
        bins=FreedmanDiaconis(unfolded),
        density=True,
        color="deepskyblue",
        histtype="step",
        label=f"unfolded spectrum L={L}",
    )
    plt.plot(
        np.linspace(min(unfolded), max(unfolded), len(p)),
        distribution(unfolded, "GOE"),
        color="grey",
        linestyle="--",
        label="GOE",
    )
    plt.legend()
    plt.show()


def plotGUE():
    pass


def plotGSE():
    pass


if __name__ == "__main__":
    plotDistribution()
    # plotGOE()
