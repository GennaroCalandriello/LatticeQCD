from turtle import color
from matplotlib.lines import lineStyles
from matplotlib.pyplot import hist
import numpy as np
import multiprocessing
from functools import partial
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from sympy import plot

#test commit
from modulo.functions import *
from modulo.stats import *

path1 = "data_analysis/"
pathData = f"data_analysis/{phase}/data/"
pathErrors = f"data_analysis/{phase}/errors/"


def make_dirs():

    if os.path.exists(pathData):
        shutil.rmtree(pathData)
    if os.path.exists(pathErrors):
        shutil.rmtree(pathErrors)

    os.makedirs(f"{pathData}")
    os.makedirs(f"{pathErrors}")


def compute_Is0(num_ev, spacing):

    less_than_s_0 = 0

    for s in spacing:
        if s <= s_0:
            less_than_s_0 += 1
    Is_0 = less_than_s_0 / num_ev

    return Is_0


def loadtxt(newresults=True, topocool=False):

    """Structure of data:
    1st number: RHMC trajectory
    2nd number: 1 non ti interessa
    3rd number: 200 = eigenvalues calculated
    4th number: 0/1 quark up or down
    4:204 numbers: eigenvalues
    204:204+22*200: IPR"""

    if newresults:
        # new results sent 15/03/2023: this is all the statistics we have
        edeconfined = np.loadtxt("PhDresults\deconf\deconfined_bz0\deconfined_bz0.txt")
        econfined = np.loadtxt("PhDresults\conf\confined_bz0\confined_bz0.txt")

    else:
        edeconfined = np.loadtxt("PhDresults/deconf/deconfined/deconfined_bz0.txt")
        econfined = np.loadtxt("PhDresults/conf/confined/confined_bz0.txt")

    if topocool:

        TOPOLOGICALconf = np.loadtxt("PhDresults\conf\confined_bz0\TopoCool.txt")
        TOPOLOGICALdeconf = np.loadtxt("PhDresults\deconf\deconfined_bz0\TopoCool.txt")

        return edeconfined, econfined, TOPOLOGICALdeconf, TOPOLOGICALconf

    else:

        return edeconfined, econfined


def loading():

    """This function load the data from the txt file and return max and min values of the spectrum"""

    edeconfined, econfined, topodec, topoconf = loadtxt(topocool=True)
    print(topodec.shape)
    edeconfined = np.abs(edeconfined[:, 4:204])
    econfined = np.abs(econfined[:, 4:204])
    econfined = econfined[:, ::2]
    edeconfined = edeconfined[:, ::2]

    configurations = len(edeconfined[:, 0])

    # print max
    maxdec = np.amax(edeconfined)
    maxconf = np.amax(econfined)
    mindec = np.amin(edeconfined)
    minconf = np.amin(econfined)

    return econfined, edeconfined, maxdec, mindec, maxconf, minconf


def loadSortRank():

    edeconfined, econfined, topodec, topoconf = loadtxt(topocool=True)
    print(topodec.shape)
    edeconfined = np.abs(edeconfined[:, 4:204])
    econfined = np.abs(econfined[:, 4:204])
    econfined = econfined[:, ::2]
    edeconfined = edeconfined[:, ::2]

    # statistical test to ensure that configurations are correctly generated
    # indicesRandom = np.random.choice(edeconfined.shape[0], 300, replace=False)
    # edeconfined = edeconfined[indicesRandom, :]

    configurations_dec = len(edeconfined[:, 0])
    configurations_conf = len(econfined[:, 0])

    # print max

    unsorted_dec = []
    unsorted_conf = []
    ranked_dec = []
    ranked_conf = []

    # 1. ordino tutti gli autovalori lambda(i,j) di tutte le configurazioni
    for i in range(len(edeconfined[:, 0])):
        for j in range(len(edeconfined[0, :])):
            unsorted_dec.append([int(i + 1), edeconfined[i, j]])

    for i in range(len(econfined[:, 0])):
        for j in range(len(econfined[0, :])):
            unsorted_conf.append([int(i + 1), econfined[i, j]])

    sorted_dec = sorted(unsorted_dec, key=lambda x: x[1])
    sorted_conf = sorted(unsorted_conf, key=lambda x: x[1])

    # rimpiazzo i lambda(i,j) con il loro rango r nell'insieme di tutti gli autovalori diviso
    # il numero di configurazioni
    rank1 = np.arange(1, len(sorted_dec[:]) + 1, 1)
    rank2 = np.arange(1, len(sorted_conf[:]) + 1, 1)
    rank1 = rank1 / configurations_dec
    rank2 = rank2 / configurations_conf

    # here in the ranked_dec I append: [configurazione, rango, lambda(i,j)]
    for k in range(len(rank1)):
        ranked_dec.append([sorted_dec[k][0], rank1[k], sorted_dec[k][1]])

    for k in range(len(rank2)):
        ranked_conf.append([sorted_conf[k][0], rank2[k], sorted_conf[k][1]])

    return ranked_dec, ranked_conf, configurations_dec, configurations_conf


def spacingCalculus(bins):

    """This function calculate the spacing distribution for each bin.
    It takes bins in input structured as follows:
    bins = [config_x: [ranked_ev, real_ev], config_y: [ranked_ev, real_ev], ...]"""

    mean_spacings = []
    Is0 = []
    Is0_with_kde = []
    errors_Is0 = []
    errors_Is0_with_kde = []
    errors_mean_spacings = []
    mean_eigenvalues = []

    # qui ci sono tutti i bins
    for b in range(len(bins) - 1):
        spacing = []
        bin = bins[b]
        bin_next = bins[b + 1]
        count_ev_in_bin = 0
        eigenvalueList = []
        for config, ranked_ev in bin.items():

            # #--------here I add the first spacing of the next bin to the previous one to minimize the partition of the spectrum in bins----------
            if (
                config in bin_next
                and len(bin_next[config]) > 0
                and len(bin[config]) > 0
            ):
                last_value = bin[config][-1][
                    0
                ]  # ok the structure is: bin[config][ranked][real_lambda]
                first_value_next = bin_next[config][0][0]
                added_s = first_value_next - last_value

            else:
                added_s = 0
            # # -------------------------------------------------------------------------------------------------------------------------------------

            for e in range(len(ranked_ev) - 1):
                spacing.append(((ranked_ev[e + 1][0] - ranked_ev[e][0])))

            for e in range(len(ranked_ev)):
                count_ev_in_bin += 1
                eigenvalueList.append(ranked_ev[e][1])

            spacing.append(added_s)

        # spacing for each bin in which the spectrum is divided
        spacing = np.array(spacing)
        eigenvalueList = np.array(eigenvalueList)

        
        # -------------------------------------------------------------------------------------

        # here I calculate the error on Is0 and on the mean spacing
        # kind= 1 for Is0, kind=2 for mean spacing
        if calculate_errors:
            errors_Is0.append(errorAnalysis(count_ev_in_bin, spacing, kind=1))
            errors_mean_spacings.append(errorAnalysis(count_ev_in_bin, spacing, kind=3))
            errors_Is0_with_kde.append(errorAnalysis(count_ev_in_bin, spacing, kind=2))
        else:
            errors_Is0.append(0)
            errors_mean_spacings.append(0)
            errors_Is0_with_kde.append(0)

        # here I calculate Is0 and Is0 with kde
        Is0.append(compute_Is0(count_ev_in_bin, spacing))
        Is0_with_kde.append(
            KernelDensityFunctionIntegrator(spacing, FreedmanDiaconis(spacing))
        )

        # print("mean spacing", np.mean(spacing))
        mean_spacings.append(np.mean(spacing))
        spacing = np.array(spacing)
        eigenvalueList = np.array(eigenvalueList)
        mean_eigenvalues.append(np.mean(eigenvalueList))

    # save the data---------------------------------------------------------------------------
    if savedata:
        # SAVE spacings for histogram plot-------------------------------------------------------
        np.savetxt(
            f"{pathData}/spacings_{round(min(eigenvalueList), 4)}-{round(max(eigenvalueList), 4)}.txt",
            spacing,
        )
        np.savetxt(f"{pathData}/Is0.txt", Is0)
        np.savetxt(f"{pathData}/mean_spacings.txt", mean_spacings)

        
        np.savetxt(f"{pathErrors}/mean_spacings_errors.txt", errors_mean_spacings)
        np.savetxt(f"{pathErrors}/Is0_errors.txt", errors_Is0)
        np.savetxt(f"{pathErrors}/Is0_with_kde_errors.txt", errors_Is0_with_kde)

        np.savetxt(f"{pathData}/mean_eigenvalues.txt", mean_eigenvalues)
        np.savetxt(f"{pathData}/Is0_with_kde.txt", Is0_with_kde)

    return Is0, Is0_with_kde, mean_spacings, errors_Is0, errors_Is0_with_kde, errors_mean_spacings, mean_eigenvalues

def mean_eigenvalues(data):
    data = data[:, ::2]
    conf = len(data)
    num_ev = len(data[0])
    mean = np.zeros(num_ev)

    for ev in range(num_ev):
        summ = sum(data[:, ev])
        mean[ev] = summ / conf

    return mean

def spectralRegionsAnalysis(spectral_window):

    """Qui seleziono un certo range e calcolo la differenza s(i,j)= x(i,j+1) - x(i,j)
    reference: Localization properties of Dirac modes at the Roberge-Weiss phase transition, PHYS. REV. D 105, 014506 (2022)"""

    # here in the ranked_dec I append: [configurazione, rango, lambda(i,j)]
    ranked_dec, ranked_conf, configurations_dec, configurations_conf = loadSortRank()

    if phase == "deconfined":
        ranked_ = ranked_dec
        configurations = configurations_dec
    elif phase == "confined":
        ranked_ = ranked_conf
        configurations = configurations_conf

    bins = []

    spectral_bins = np.arange(mindec, maxdec, spectral_window)

    # Split the spectrum in bins
    for s in range(len(spectral_bins) - 1):
        bins.append(
            [
                ranked_[i]
                for i in range(len(ranked_))
                if spectral_bins[s] <= ranked_[i][2] < spectral_bins[s + 1]
            ]
        )

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

    Is0, Is0_with_kde, _, _, _, _, lambda_values = spacingCalculus(grouped_bins)

    Is0=np.array(Is0)

    Is0_with_kde=np.array(Is0_with_kde)
    lambda_values=np.array(lambda_values)

    if lambda_edge_with_KDE:
        lambda_edge_via_Is0(Is0_with_kde, lambda_values)

    else:
        lambda_edge_via_Is0(Is0, lambda_values)


def plot_ULSD():
    """Plot the unfolded level spacing distribution"""
    
    s1 = np.loadtxt(f"{pathData}/spacings_0.0122-0.0152.txt")
    Poisson = distribution(s1, "Poisson")
    x = np.linspace(min(s1), max(s1), len(s1))

    plt.figure()
    plt.hist(s1, bins=FreedmanDiaconis(s1), density=True, label="Data", histtype="step")
    plt.plot(x, Poisson, label="Poisson", color="red")
    plt.show()

def lambda_edge_via_Is0(Is0, lambda_values):

        method = 1

        # Find the closest point to Is0cr

        # approx_lambda_c = lambda_values[np.abs(Is0 - Is0crit).argmin()]
        # print("approximate lambda_c", approx_lambda_c)
        # Find the closest point to Is0cr
        if method == 1:
            approx_lambda_c = lambda_values[np.abs(Is0 - Is0crit).argmin()]

            # Consider points around approx_lambda_c for linear fit
            window_size = 0.01  # Define window size
            mask = (lambda_values > approx_lambda_c - window_size) & (
                lambda_values < approx_lambda_c + window_size
            )

            # Linear fit function
            def linear(x, a, b):
                return a * x + b

            popt, _ = curve_fit(linear, lambda_values[mask], Is0[mask])

            # Find lambda_c where Is0(λ) = Is0cr
            lambda_c = (Is0crit - popt[1]) / popt[0]

            print("Determined lambda_c:", lambda_c)

def mobility_edge():

    """Calculate the mobility edge through Is0 using various spectral regions"""
    DeltaLambda =np.linspace(0.001, 0.006, 30) #here various spectral regions
    for deltaL in DeltaLambda:
        print("spectral window: ", deltaL)
        spectralRegionsAnalysis(deltaL)


if __name__ == "__main__":
    make_dirs()
    mobility_edge()
  

    pass
