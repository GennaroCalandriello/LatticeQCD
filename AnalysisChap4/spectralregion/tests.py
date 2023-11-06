from hmac import new
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from collections import defaultdict

nbins = 30


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
    maxdec = np.amax(edeconfined)
    maxconf = np.amax(econfined)
    mindec = np.amin(edeconfined)
    minconf = np.amin(econfined)

    print("maxdeconfined", maxdec)
    print("maxconfined", maxconf)

    unsorted_dec = []
    unsorted_conf = []
    ranked_dec = []
    ranked_conf = []
    print(maxdec)
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

    return (
        sorted_dec,
        sorted_conf,
        ranked_dec,
        ranked_conf,
        configurations_dec,
        configurations_conf,
    )

    # u = unfold_spectrum_per_config(sorted_dec)

    # rimpiazzo i lambda(i,j) con il loro rango r nell'insieme di tutti gli autovalori diviso
    # il numero di configurazioni


def unfold_spectrum_per_config(eigenvalues_with_config):
    # Separate eigenvalues and their configurations
    # Experimental function to unfold the spectrum
    configs, eigenvalues = zip(*eigenvalues_with_config)
    nconf = max(configs)
    print(nconf)
    # Generate the staircase function N(E) for each configuration
    E = np.array(eigenvalues)
    N = np.arange(1, len(E) + 1) / nconf

    coeff = np.polyfit(E, N, 100)
    p = np.poly1d(coeff)
    dense_lambda = np.linspace(min(E), max(E), len(E))
    dense_N = abs(p(dense_lambda))
    new_spectrum = []

    # plt.figure()
    # plt.plot(dense_lambda, dense_N)
    # plt.plot(E, N)
    # plt.show()
    for i in range(len(eigenvalues)):
        new_spectrum.append([configs[i], dense_N[i], eigenvalues[i]])

    return new_spectrum


def prova():

    sorted_dec, _, _, _, conf, _ = loadSortRank()
    ranked_dec = unfold_spectrum_per_config(sorted_dec)
    # 1. Grouping the data by configuration

    # grouped_data = defaultdict(list)
    # for entry in ranked_dec:
    #     configuration = entry[0]
    #     grouped_data[configuration].append(entry)

    # # 2. Sorting within each group by the unfolded eigenvalue
    # for configuration, entries in grouped_data.items():
    #     grouped_data[configuration] = sorted(entries, key=lambda x: x[2])

    # # 3. Calculating the spacing (difference)
    # spacing_data = {}
    # for configuration, sorted_entries in grouped_data.items():
    #     spacings = []
    #     for i in range(1, len(sorted_entries)):

    #         spacing = sorted_entries[i][2] - sorted_entries[i - 1][2]
    #         spacings.append(spacing)
    #     spacing_data[configuration] = spacings

    # 4. Instead of grouping by configuration, divide the spectrum in nbins bins
    bin_width = len(ranked_dec) // nbins
    bin_data = defaultdict(list)

    for i in range(nbins):
        bin_data[i] = ranked_dec[i * bin_width : (i + 1) * bin_width]

    # 5. Calculate the spacing for each bin
    spacing_bin_data = {}
    for bin, entries in bin_data.items():
        # 6. group by configuration in each bin
        grouped_data = defaultdict(list)
        for entry in entries:
            configuration = entry[0]
            grouped_data[configuration].append(entry)
        # 7. sort by unfolded eigenvalue
        for configuration, entries in grouped_data.items():
            grouped_data[configuration] = sorted(entries, key=lambda x: x[2])
        # 8. calculate spacing
        spacings = []
        for configuration, sorted_entries in grouped_data.items():
            for i in range(1, len(sorted_entries)):
                spacing = sorted_entries[i][2] - sorted_entries[i - 1][2]
                spacings.append(spacing)
        spacing_bin_data[bin] = spacings
        print(np.mean(spacings))


if __name__ == "__main__":
    prova()
