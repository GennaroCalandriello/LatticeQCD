import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import math

'''This file contains functions that perform unfolding through CDF for each single configuration. MainUnfold2 selects the
range of eigenvalues for each configuration. This approach is not appropriate if the statistics is low'''

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
            p[i] = (32 / np.pi ** 2) * s[i] ** 2 * np.exp(-4 / np.pi * s[i] ** 2)

    if kind == "GSE":
        for i in range(len(p)):
            p[i] = (
                2 ** 18
                / (3 ** 6 * np.pi ** 3)
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

def unfold_spectrum(eigenvalues):

    bins = int(1.3 * np.log2(len(eigenvalues) + 1))
    hist, bin_edges = np.histogram(eigenvalues, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cdf = np.cumsum(hist) / np.sum(hist)

    cdf_spline = InterpolatedUnivariateSpline(bin_centers, cdf, k=1)

    unfolded_eigenvalues = cdf_spline(eigenvalues)

    return unfolded_eigenvalues

def level_spacing(unfolded_ev):

    sorted_unfolded_ev = np.sort(unfolded_ev)
    spacings = np.diff(sorted_unfolded_ev)

    return spacings / np.mean(spacings)


def MainUnfold(data, rng, plot=True):

    """rng selects the range of \lambda lt and gt certain values"""

    print("Exe in range:", rng)
    low = rng[0]
    high = rng[1]
    data = np.abs(data[:, 4:204])
    data = np.array(data[:, ::2])
    N_conf = len(data[:])

    spacing_list = []
    data_ranged = []

    for i in range(N_conf):
        temp = []
        for j in data[i]:
            if j <= high and j >= low:
                temp.append(j)
        data_ranged.extend([temp])
    data_ranged = np.array(data_ranged)
    print("data last", data_ranged[0])

    for i in range(N_conf):
        ev_temp = []
        for ev in data_ranged[i]:
            ev_temp.append(ev)

        unfolded = unfold_spectrum(ev_temp)
        spacings = level_spacing(unfolded)
        spacing_list.extend(spacings)

    spacing_list = np.array(spacing_list)

    if plot:
        GUE = distribution(spacing_list, "GUE")
        Poisson = distribution(spacing_list, "Poisson")
        s = np.linspace(min(spacing_list), max(spacing_list), len(spacing_list))
        bins = 40  # int(1.4 * (np.log2(len(spacing_list) + 1)))
        plt.figure()
        plt.plot(s, GUE, "g--", label="GUE")
        plt.plot(s, Poisson, "b--", label=Poisson)
        plt.hist(spacing_list, bins=bins, density=True, histtype="step", color="red")
        plt.show()

    return spacing_list


def MainUnfold2(data, rng):
    """rng selects the int range of eigenvalues that you want for all configurations (first 5, next 5 etc...)"""

    plot = True

    print("Exe in range:", rng)
    low = rng[0]
    high = rng[1]
    data = np.abs(data[:, 4:204])
    data = np.array(data[:, ::2])
    data = data[:, low:high]
    N_conf = len(data[:])

    spacing_list = []
    data_ranged = []

    for i in range(N_conf):
        ev_temp = []
        for ev in data[i]:
            ev_temp.append(ev)
        unfolded = unfold_spectrum(ev_temp)
        spacings = level_spacing(unfolded)
        spacing_list.extend(spacings)

    spacing_list = np.array(spacing_list)

    GUE = distribution(spacing_list, "GUE")
    Poisson = distribution(spacing_list, "Poisson")
    s = np.linspace(min(spacing_list), max(spacing_list), len(spacing_list))
    bins = 40  # int(1.4 * (np.log2(len(spacing_list) + 1)))

    if plot:
        plt.figure()
        plt.plot(s, GUE, "g--", label="GUE")
        plt.plot(s, Poisson, "b--", label=Poisson)
        plt.hist(spacing_list, bins=bins, density=True, histtype="step", color="red")
        plt.show()

    return spacing_list