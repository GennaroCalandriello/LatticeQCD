from ast import Is
import random
import re
from turtle import color
from venv import create
import numpy as np
import multiprocessing
from functools import partial
from collections import defaultdict

from main import *
from modulo.functions import *
from modulo.stats import *

num_bins = 40


def loadSortRank():

    edeconfined, econfined = loadtxt()
    edeconfined = np.abs(edeconfined[:, 4:204])
    econfined = np.abs(econfined[:, 4:204])
    econfined = econfined[:, ::2]
    edeconfined = edeconfined[:, ::2]

    # statistical test to ensure that configurations are correctly generated
    # indicesRandom = np.random.choice(edeconfined.shape[0], 300, replace=False)
    # edeconfined = edeconfined[indicesRandom, :]

    configurations = len(edeconfined[:, 0])

    # print max
    print("Max deconfined: ", np.amax(edeconfined))
    print("Max confined: ", np.amax(econfined))
    unsorted_dec = []
    unsorted_conf = []
    ranked_dec = []
    ranked_conf = []

    # 1. ordino tutti gli autovalori lambda(i,j) di tutte le configurazioni
    for i in range(len(edeconfined[:, 0])):
        for j in range(len(edeconfined[0, :])):
            unsorted_dec.append([int(i + 1), edeconfined[i, j]])
            unsorted_conf.append([int(i + 1), econfined[i, j]])

    sorted_dec = sorted(unsorted_dec, key=lambda x: x[1])
    sorted_conf = sorted(unsorted_conf, key=lambda x: x[1])

    # rimpiazzo i lambda(i,j) con il loro rango r nell'insieme di tutti gli autovalori diviso
    # il numero di configurazioni
    rank = np.arange(1, len(sorted_dec[:]) + 1, 1)
    rank = rank / configurations

    # here in the ranked_dec I append: [configurazione, rango, lambda(i,j)]
    for k in range(len(rank)):
        ranked_dec.append([sorted_dec[k][0], rank[k], sorted_dec[k][1]])
        ranked_conf.append([sorted_conf[k][0], rank[k], sorted_conf[k][1]])

    return ranked_dec, ranked_conf, configurations


def spacingCalculus(bins):

    """This function calculate the spacing distribution for each bin.
    It takes bins in input structured as follows:
    bins = [config_x: [ranked_ev, real_ev], config_y: [ranked_ev, real_ev], ...]"""

    plotto = False
    mean_spacings = []
    IS0 = []

    # qui ci sono tutti i bins
    for b in range(len(bins) - 1):
        spacing = []
        bin = bins[b]
        bin_next = bins[b + 1]

        for config, ranked_ev in bin.items():
            if config in bin_next:
                last_value = bin[config][-1][
                    0
                ]  # ok the structure is: bin[config][ranked][real_lambda]
                first_value_next = bin_next[config][0][0]
                added_s = first_value_next - last_value
            else:
                added_s = 0

            for e in range(len(ranked_ev) - 1):
                spacing.append((ranked_ev[e + 1][0] - ranked_ev[e][0]))

            spacing.append(added_s)

        print("mean spacing", np.mean(spacing))
        mean_spacings.append(np.mean(spacing))
        spacing = np.array(spacing)
        is0 = KernelDensityFunctionIntegrator(
            spacing, FreedmanDiaconis(spacing), plot=False
        )
        IS0.append(is0)

        if plotto:
            spacing = np.array(spacing)
            plot = np.linspace(min(spacing), max(spacing), len(spacing))
            Poisson = distribution(spacing, "Poisson")
            GUE = distribution(plot, "GUE")
            plt.figure()
            plt.hist(
                spacing,
                bins=FreedmanDiaconis(spacing),
                density=True,
                histtype="step",
            )
            plt.legend()
            plt.plot(plot, Poisson, "g--")
            plt.plot(plot, GUE, "r--")
            plt.xlabel("s")
            plt.ylabel("P(s)")
            plt.show()

    # here I plot the mean spacing for each bin (validation for unfolding: <s> = 1)
    plt.figure()
    plt.scatter(
        np.arange(1, len(mean_spacings) + 1, 1), mean_spacings, marker="+", color="blue"
    )
    plt.axhline(y=1, color="r", linestyle="--")
    plt.show()

    # here I plot the IS0 for each bin ma non va devo riscriverlo
    plt.figure()
    plt.scatter(range(len(IS0)), IS0, marker="+", color="blue")
    plt.show()


def selectRangeandPlot():

    """Qui seleziono un certo range e calcolo la differenza s(i,j)= x(i,j+1) - x(i,j)
    reference: Localization properties of Dirac modes at the Roberge-Weiss phase transition, PHYS. REV. D 105, 014506 (2022)"""

    # here in the ranked_dec I append: [configurazione, rango, lambda(i,j)]
    ranked_dec, ranked_conf, configurations = loadSortRank()

    bin_size = len(ranked_dec) // num_bins
    print("size", bin_size)
    bins = []

    # Split the spectrum in bins
    for i in range(0, len(ranked_dec), bin_size):
        bins.append(ranked_dec[i : i + bin_size])

    grouped_bins = []

    # Loop through all bins
    for b in bins:

        # Initialize a defaultdict to store eigenvalues of the same configuration
        grouped = defaultdict(list)

        # Loop through each [config, eigenvalue] pair in the bin
        for config, ranked_lambda, real_lambda in b:
            # Append the eigenvalue to the corresponding configuration
            grouped[config].append([ranked_lambda, real_lambda])

        # Convert the defaultdict to a regular dictionary and append it to the list
        grouped_bins.append((dict(grouped)))

    spacingCalculus(grouped_bins)


if __name__ == "__main__":
    selectRangeandPlot()
