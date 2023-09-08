from venv import create
import numpy as np
import multiprocessing
from functools import partial
from collections import defaultdict

from main import *
from modulo.functions import *
from modulo.stats import *


def loadSortRank():

    edeconfined, econfined = loadtxt()
    edeconfined = edeconfined[:, 4:204]
    econfined = econfined[:, 4:204]
    configurations = len(edeconfined[:, 0])

    # print max
    unsorted_dec = []
    unsorted_conf = []
    ranked_dec = []
    ranked_conf = []

    # 1. ordino tutti gli autovalori lambda(i,j) di tutte le configurazioni
    for i in range(len(edeconfined[:, 0])):
        for j in range(len(edeconfined[0, :])):
            if edeconfined[i, j] >= 0:
                unsorted_dec.append([int(i), edeconfined[i, j]])
                unsorted_conf.append([int(i), econfined[i, j]])

    sorted_dec = sorted(unsorted_dec, key=lambda x: x[1])
    sorted_conf = sorted(unsorted_conf, key=lambda x: x[1])

    # rimpiazzo i lambda(i,j) con il loro rango r nell'insieme di tutti gli autovalori diviso
    # il numero di configurazioni
    rank = np.arange(0, len(sorted_dec[:]), 1)
    rank = rank / configurations

    # here in the ranked_dec I append: [configurazione, rango, lambda(i,j)]
    for k in range(len(rank)):
        ranked_dec.append([sorted_dec[k][0], rank[k], sorted_dec[k][1]])
        ranked_conf.append([sorted_conf[k][0], rank[k], sorted_conf[k][1]])

    return ranked_dec, ranked_conf


def selectRangeandPlot(rng, kind="dec"):
    print("Range: ", rng)

    """Qui seleziono un certo range e calcolo la differenza s(i,j)= x(i,j+1) - x(i,j)
    reference: Localization properties of Dirac modes at the Roberge-Weiss phase transition, PHYS. REV. D 105, 014506 (2022)"""

    ranked_dec, ranked_conf = loadSortRank()
    low = rng[0]
    high = rng[1]
    eigenvalues_dec = []
    eigenvalues_conf = []
    for i in range(len(ranked_dec)):
        if ranked_dec[i][2] >= low and ranked_dec[i][2] <= high:
            eigenvalues_dec.append([ranked_dec[i][0], ranked_dec[i][1]])
            eigenvalues_conf.append([ranked_conf[i][0], ranked_conf[i][1]])

    # Group the eigenvalues by their configuration
    eigenvalues_by_config_dec = defaultdict(list)
    eigenvalues_by_config_conf = defaultdict(list)

    for config, eigenvalue in eigenvalues_dec:
        eigenvalues_by_config_dec[config].append(eigenvalue)

    for config, eigenvalue in eigenvalues_conf:
        eigenvalues_by_config_conf[config].append(eigenvalue)

    # Sort the eigenvalues for each configuration and calculate the differences
    differences_by_config = {}
    diff_dec = []
    diff_conf = []

    # here for deconfined
    for config, eigenvalues in eigenvalues_by_config_dec.items():
        eigenvalues = sorted(eigenvalues)  # Sort the eigenvalues
        differences = []
        for i in range(len(eigenvalues) - 1):
            differences.append(eigenvalues[i] * (eigenvalues[i + 1] - eigenvalues[i]))

        differences = np.array(differences) / np.mean(eigenvalues)

        # differences = np.diff(eigenvalues)  # Calculate the differences between adjacent eigenvalues
        differences_by_config[config] = differences.tolist()  # Store the differences
        diff_dec.extend(differences.tolist())

    eigenvalues = 0
    differences = 0

    # here for confined
    for config, eigenvalues in eigenvalues_by_config_conf.items():

        eigenvalues = sorted(eigenvalues)  # Sort the eigenvalues
        differences = []
        for i in range(1, len(eigenvalues)):
            differences.append(eigenvalues[i] * (eigenvalues[i] - eigenvalues[i - 1]))
        # for i in range(1, len(differences)):
        #     differences[i] = differences[i] * eigenvalues[i - 1]

        differences = np.array(differences) / np.mean(eigenvalues)
        differences_by_config[config] = differences.tolist()
        diff_conf.extend(differences.tolist())

    diff_conf = np.sort(diff_conf)
    diff_dec = np.sort(diff_dec)

    # select what do you want to plot
    if kind == "dec":
        spacing = diff_dec
        kindplot = "Deconfined eigenvalues"
    elif kind == "conf":
        spacing = diff_conf
        kindplot = "Confined eigenvalues"

    # plot the histogram
    plot = np.linspace(0, max(spacing), len(spacing))
    Poisson = distribution(plot, "Poisson")
    GUE = distribution(plot, "GUE")

    plt.figure()
    plt.hist(
        spacing,
        bins=FreedmanDiaconis(spacing),
        density=True,
        histtype="step",
        label=r"range $\lambda$:" f"[{low},{high}]",
    )
    plt.legend()
    plt.plot(plot, Poisson, "g--")
    plt.plot(plot, GUE, "r--")
    plt.title(f"{kindplot}")
    plt.xlabel("s")
    plt.ylabel("P(s)")
    plt.show()


def selectRange(num):
    edeconfined, econfined = loadtxt()
    edeconfined = np.abs(edeconfined[:, 4:204])
    econfined = np.abs(econfined[:, 4:204])
    econfined = econfined[:, ::2]
    edeconfined = edeconfined[:, ::2]

    # prendo i valori max e min tra tutti gli autovalori
    max_confined = np.amax(econfined)
    max_deconfined = np.amax(edeconfined)
    min_confined = np.amin(econfined)
    min_deconfined = np.amin(edeconfined)

    rangeConfined = CreateRanges(num, min_confined, max_confined)
    rangeDeconfined = CreateRanges(num, min_deconfined, max_deconfined)

    return rangeConfined, rangeDeconfined


if __name__ == "__main__":
    # selectRange()
    # rngConfined, rngDeconfined = selectRange(num=4)
    rngDeconfined = [
        [0.0, 0.01],
        [0.01, 0.015],
        [0.015, 0.025],
        [0.025, 0.035],
        [0.035, 0.045],
    ]
    # with multiprocessing.Pool(processes=len(rngDeconfined)) as pool:
    #     result = np.array(pool.map(selectRangeandPlot, rngDeconfined))
    #     pool.close()
    #     pool.join()
    selectRangeandPlot([0.025, 0.036])


# Unfolding algorithm:
#
# 1. ordino tutti gli autovalori lambda(i,j) di tutte le configurazioni - i è la configurazione,
#    j è il numero dell'autovalore (ordinati per grandezza per ogni configurazione)
# 2. rimpiazzo i lambda(i,j) con il loro rango r nell'insieme di tutti gli autovalori diviso
#    il numero di configurazioni - questi sono gli autovalori unfolded, x(i,j) = r(i,j)/n. conf.
# 3. calcolo la differenza s(i,j)= x(i,j+1) - x(i,j) - questi sono gli unfolded spacing
