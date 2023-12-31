import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import shutil
from modulo.par import *

from modulo.stats import FreedmanDiaconis


degrees = 3  # must be 1<= k <=5


def compute_Is0(num_ev, spacing):

    less_than_s_0 = 0

    for s in spacing:
        if s <= s_0:
            less_than_s_0 += 1
    Is_0 = less_than_s_0 / num_ev

    return Is_0


def get_key(x):
    return x[0]


def numba_sorted(arr):

    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if get_key(arr[j]) < get_key(arr[min_idx]):
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr


def sorting_and_ranking(eigenvalues):

    sorted_eigenvalues = sorted(eigenvalues, key=lambda x: x[0])
    # sorted_eigenvalues = numba_sorted(eigenvalues)
    rank = np.arange(0, len(sorted_eigenvalues[:]), 1)
    sorted_eigenvalues = np.array(sorted_eigenvalues)
    rank_list = []

    for i in range(len(rank)):
        rank_list.append([rank[i], (sorted_eigenvalues[i, 1])])

    return rank_list


def spacing_evaluation(ranked_ev):

    ranked_ev = np.array(ranked_ev)
    N_conf = round(max(ranked_ev[:, 1]))  # number of configurations

    ranked_ordered = sorted(ranked_ev, key=lambda x: x[1])
    ranked_ordered = np.array(ranked_ordered)
    spacing = []
    for j in range(N_conf + 1):
        temp = []
        spac = []

        for i in range(len(ranked_ev)):
            if ranked_ev[i, 1] == j:
                temp.append(ranked_ev[i, 0])
        for i in range(0, len(temp) - 1):
            # temp1.append(min(temp[i + 1] - temp[i], temp[i] - temp[i - 1]))
            spac.append(temp[i + 1] - temp[i])
            # spacing.append(temp[i + 1] - temp[i])
        # spac=spac/np.mean(spac)
        spacing.extend(spac)
        # temp1 = temp1 / np.mean(temp1)
    spacing = np.array((spacing)) / N_conf
    spacing = spacing / np.mean((spacing))

    return spacing


def spacing_(ranked_ev):

    ranked_ordered = sorted(ranked_ev, key=lambda x: x[1])
    ranked_ordered = np.array(ranked_ordered)
    N_conf = round(max(ranked_ordered[:, 1]))  # number of configurations
    spacings = []
    temp = []

    for s in range(len(ranked_ordered)):
        spacings.append(ranked_ordered[s, 0])

    spacings = np.diff((spacings))
    spacings = np.abs(spacings)
    spacings = spacings / (np.mean(spacings) * N_conf)

    # spacings = spacings / np.mean((spacings))

    return spacings


def binning(eigenvalues, maxbins):

    bins = np.linspace(min(eigenvalues), max(eigenvalues), maxbins)
    eigenvalues = np.array(eigenvalues)
    digitized = np.digitize(eigenvalues, bins)
    binned_data = [eigenvalues[digitized == i] for i in range(1, len(bins))]

    return binned_data


def Tuning(spacings):

    length = len(spacings)
    print("lunghezza", length)
    row = int(length / 5)
    bins = np.zeros((row, 5))

    for i in range(0, length - 5, 5):
        k = int(i / 5)
        for j in range(5):
            if (spacings[i + j] - spacings[i]) < (spacings[i + 4] - spacings[i + j]):
                bins[k, j] = spacings[i]
            else:
                bins[k, j] = spacings[i + 4]
    # print("Bin Boundaries: \n", bins)

    new_spacings = []
    for i in range(len(bins)):
        new_spacings.extend(bins[i])

    return new_spacings


def histogramFunction(sp, kind: str, bins, low, high):

    GUE = distribution(sp, "GUE")
    GSE = distribution(sp, "GSE")
    GOE = distribution(sp, "GOE")
    POISSON = distribution(sp, "Poisson")
    x = np.linspace(min(sp), max(sp), len(sp))

    std = np.std(sp)
    plt.figure()
    plt.xlabel("s", fontsize=15)
    plt.ylabel("P(s)", fontsize=15)
    plt.title(f"{kind}", fontsize=18)
    binned = np.array(binning(sp, bins + 1))
    errori = []

    for i in range(len(binned)):
        # length = len(binned)
        # std_err = 1 / (np.sqrt(length))*np.std(binned[i])
        std_err = stats.sem(binned[i])
        errori.append(std_err)

    counts, edges, _ = plt.hist(
        sp,
        bins=bins,
        density=True,
        histtype="step",
        fill=False,
        color="blue",
        label=r"$\lambda \in$" f"{round(low, 4)}; {round(high, 4)}",
    )
    # print("mi printi questo counts?", counts)
    # print("e mo printami i dati binnati", binned)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    # bin_width = edges[1] - edges[0]
    plt.errorbar(bin_centers, counts, yerr=errori, xerr=0, fmt=".", color="blue")

    plt.plot(x, GUE, "g--", label="GUE")
    # plt.plot(x, GSE, "b--", label="GSE")
    plt.plot(x, GOE, "r--", label="GOE")
    plt.plot(x, POISSON, "y--", label="Poisson")
    plt.legend()
    plt.show()


def reshape_to_matrix(array):

    shape = (round(len(array) / 100), 100)
    rows, cols = shape
    return [[array[i * cols + j] for j in range(cols)] for i in range(rows)]


def distribution(sp, kind):

    """Plot theoretical distributions of GSE, GOE, GUE ensemble distributions picking the min and max values of the spacing array
    calculated in the main program"""
    s = np.linspace(0, max(sp), len(sp))
    p = np.zeros(len(s))

    if kind == "GOE":
        for i in range(len(p)):
            p[i] = np.pi / 2 * s[i] * np.exp(-np.pi / 4 * s[i] ** 2)

    if kind == "GUE":
        for i in range(len(p)):
            p[i] = (32 / np.pi**2) * s[i] ** 2 * np.exp(-4 / np.pi * s[i] ** 2)

    if kind == "GSE":
        for i in range(len(p)):
            p[i] = (
                2**18
                / (3**6 * np.pi**3)
                * s[i] ** 4
                * np.exp(-(64 / (9 * np.pi)) * s[i] ** 2)
            )
    if kind == "Poisson":
        for i in range(len(p)):
            p[i] = np.exp(-s[i])

    if kind == "GOE FN":  # lasciamo perdere va...

        a = (27 / 8) * np.pi
        for i in range(len(p)):
            p[i] = (
                (a / np.pi)
                * s[i]
                * np.exp(-2 * a * s[i] ** 2)
                * (
                    np.pi
                    * np.exp((3 * a / 2) * s[i] ** 2)
                    * (a * s[i] ** 2 - 3)
                    * (
                        math.erf(np.sqrt(a / 6) * s[i])
                        - math.erf(np.sqrt(3 * a / 2) * s[i])
                        + np.sqrt(6 * np.pi * a)
                        * s[i]
                        * (np.exp((4 * a / 3) * s[i] ** 2) - 3)
                    )
                )
            )

    return p
