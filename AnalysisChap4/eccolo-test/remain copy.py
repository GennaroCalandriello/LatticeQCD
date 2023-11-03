import numpy as np
import multiprocessing
from functools import partial
from collections import defaultdict

from modulo.functions import *
from modulo.stats import *

path1 = "data_analysis/"
pathData = f"data_analysis/{phase}/data/"
pathErrors = f"data_analysis/{phase}/errors/"


def make_dirs():

    if os.path.exists(path1):
        shutil.rmtree(path1)

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

    print("maxdeconfined", maxdec)
    print("maxconfined", maxconf)

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

    configurations = len(edeconfined[:, 0])

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

    mean_spacings = []
    Is0 = []
    Is0_with_kde = []
    errors_Is0 = []
    errors_Is0_with_kde = []
    errors_mean_spacings = []
    mean_eigenvalues = []
    _, _, maxdec, mindec, maxconf, minconf = loading()

    # qui ci sono tutti i bins
    for b in range(len(bins) - 1):
        spacing = []
        bin = bins[b]
        bin_next = bins[b + 1]
        count_ev_in_bin = 0
        eigenvalueList = []
        for config, ranked_ev in bin.items():

            # --------here I add the first spacing of the next bin to the previous one to minimize the partition of the spectrum in bins----------
            # if (
            #     config in bin_next
            #     and len(bin_next[config]) > 0
            #     and len(bin[config]) > 0
            # ):
            #     last_value = bin[config][-1][
            #         0
            #     ]  # ok the structure is: bin[config][ranked][real_lambda]
            #     first_value_next = bin_next[config][0][0]
            #     added_s = first_value_next - last_value

            # else:
            #     added_s = 0
            # -------------------------------------------------------------------------------------------------------------------------------------

            for e in range(len(ranked_ev) - 1):
                count_ev_in_bin += 1
                spacing.append(((ranked_ev[e + 1][0] - ranked_ev[e][0])))
                eigenvalueList.append(ranked_ev[e][1])
            count_ev_in_bin += 1
            # spacing.append(added_s)

        # spacing for each bin in which the spectrum is divided
        spacing = np.array(spacing)
        eigenvalueList = np.array(eigenvalueList)

        # SAVE spacings for histogram plot-------------------------------------------------------
        np.savetxt(
            f"{pathData}/spacings_{round(min(eigenvalueList), 4)}-{round(max(eigenvalueList), 4)}.txt",
            spacing,
        )
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
    np.savetxt(f"{pathData}/Is0.txt", Is0)
    np.savetxt(f"{pathData}/mean_spacings.txt", mean_spacings)

    if calculate_errors:
        np.savetxt(f"{pathErrors}/mean_spacings_errors.txt", errors_mean_spacings)
        np.savetxt(f"{pathErrors}/Is0_errors.txt", errors_Is0)
        np.savetxt(f"{pathErrors}/Is0_with_kde_errors.txt", errors_Is0_with_kde)

    np.savetxt(f"{pathData}/mean_eigenvalues.txt", mean_eigenvalues)
    np.savetxt(f"{pathData}/Is0_with_kde.txt", Is0_with_kde)


def spectralRegionsAnalysis():

    """Qui seleziono un certo range e calcolo la differenza s(i,j)= x(i,j+1) - x(i,j)
    reference: Localization properties of Dirac modes at the Roberge-Weiss phase transition, PHYS. REV. D 105, 014506 (2022)"""

    print("Freedman-Diaconis number of bins: ", num_bins)

    # here in the ranked_dec I append: [configurazione, rango, lambda(i,j)]
    ranked_dec, ranked_conf, configurations = loadSortRank()

    if phase == "deconfined":
        ranked_ = ranked_dec
    elif phase == "confined":
        ranked_ = ranked_conf

    bin_size = len(ranked_) // num_bins
    print("size", bin_size)
    bins = []

    # Split the spectrum in bins
    for i in range(0, len(ranked_), bin_size):
        bins.append(ranked_[i : i + bin_size])

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


def topological_charge(kind="deconfined"):
    d, c, topodec, topoconf = loadtxt(topocool=True)

    # caso deconfinato: sono 16 cariche topologiche per ogni configurazione (5312/332 = 16)
    if kind == "confined":
        topo = topoconf
    elif kind == "deconfined":
        topo = topodec
    print(topo.shape)
    topotot = []
    print(len(topo[:, 2]))
    averages = []
    _ = CDF(topo[:, 2])

    for i in range(0, len(topo[:, 2])):
        avg = round(topo[i, 2])
        averages.append(avg)
    averages = np.array(averages)

    plt.figure()
    plt.xlabel("Q")
    plt.ylabel("P(Q)")
    plt.hist(
        topo[:, 2],
        bins=5 * FreedmanDiaconis(topo[:, 2]),
        label="deconfined",
        histtype="step",
    )
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(range(len(topo[:32, 0])), topo[:32, 1], "x")
    plt.show()
    chi = np.mean(topo[:, 2] ** 2) / (36**3 * 22)
    print("chi", chi ** (1 / 4))


def mean_eigenvalues(data):
    data = data[:, ::2]
    conf = len(data)
    num_ev = len(data[0])
    mean = np.zeros(num_ev)

    for ev in range(num_ev):
        summ = sum(data[:, ev])
        mean[ev] = summ / conf

    return mean


def IPR_and_PR():
    # load the data (remember that IPR^(-1) \approx Ns^3*PR)
    Nt = 22
    Ns = 36  # questo devo chiederlo a Francesco D'Angelo
    ev = 200
    positive_ev = 100

    (
        d,
        c,
    ) = loadtxt(topocool=False)

    configurations = len(d[:, 0])
    # organize the data
    if phase == "deconfined":
        d = d
    elif phase == "confined":
        d = c
    maxdec = np.amax(np.abs(d[:, 4:204]))
    mindec = np.amin(np.abs(d[:, 4:204]))

    mean_eigenvalues_ = mean_eigenvalues(
        np.abs(d[:, 4:204])
    )  # here the mean on all configurations of all \lambda
    ipr_ = d[:, 204 : 22 * ev + 204]

    new_ipr = []

    for conf in ipr_:
        # here I take the IPR value only for positive eigenvalues
        new_data = []
        i = 0
        while i < len(conf):
            new_data.extend(conf[i : i + Nt])  # Take Nt elements
            i += 2 * Nt  # Jump Nt elements
        new_ipr.append(new_data)

    # -------------------------------calculate errors:----------------------------------------------
    if calculate_errors:

        new_ipr2 = np.zeros((len(ipr_), positive_ev, Nt))
        for c in range(len(new_ipr)):
            for e in range(positive_ev):
                new_ipr2[c][e][:] = new_ipr[c][e * Nt : (e + 1) * Nt]

        ipr_errors = []
        for e in range(positive_ev):
            print("calculate errors for eigenvalues #:", e)
            errors_temp = []
            for c in range(configurations):
                for t in range(Nt):
                    errors_temp.append(new_ipr2[c][e][t])
            errors_temp = np.array(errors_temp)
            ipr_errors.append(errorAnalysis(None, errors_temp, kind=3))

        ipr_errors = np.array(ipr_errors)

    else:
        ipr_errors = np.zeros(100)
    # ------------------------------------------------------------------------------------------------

    # mean over all eigenvalues for the same time slice, it should result in a total of 22 IPRs
    new_ipr = np.array(new_ipr)

    ipr_sum = np.zeros((len(new_ipr), positive_ev))
    # here I sum over all time slices for each \lam for each configuration

    for i in range(len(ipr_)):
        for j in range(positive_ev):
            ipr_sum[i][j] = sum(new_ipr[i][j * Nt : (j + 1) * Nt])

    ipr_mean = np.zeros(positive_ev)

    for e in range(positive_ev):
        ipr_mean[e] = np.mean(ipr_sum[:, e])

    ipr = ipr_mean

    pr = Volume / ipr
    pr_errors = ipr_errors / (
        ipr**2
    )  ##error prop.è giusto???? qua non so, è da chiedere

    # SAVE DATA
    np.savetxt(f"{pathData}/ipr_{phase}.txt", ipr)
    np.savetxt(f"{pathData}/pr_{phase}.txt", pr)
    np.savetxt(f"{pathData}/mean_eigenvalues_{phase}.txt", mean_eigenvalues_)

    if calculate_errors:
        np.savetxt(f"{pathErrors}/ipr_errors_{phase}.txt", ipr_errors)
        np.savetxt(f"{pathErrors}/pr_errors_{phase}.txt", pr_errors)


def fake_ensembles():
    dec, conf = loadtxt(topocool=False)
    pass


def SpatialExt():
    """This function calculate the spatial extension of the eigenmodes.
    references:
    [1] Anderson Localization in Quark-Gluon Plasma - Kovacs, Pittler, equation (2)"""
    pass


def plotdata():

    Is0_with_kde = np.loadtxt(f"{pathData}/Is0_with_kde.txt")
    Is0 = np.loadtxt(f"{pathData}/Is0.txt")
    mean_eigenvalues = np.loadtxt(f"{pathData}/mean_eigenvalues.txt")
    mean_spacings = np.loadtxt(f"{pathData}/mean_spacings.txt")
    IPR = np.loadtxt(f"{pathData}/ipr_{phase}.txt")
    PR = np.loadtxt(f"{pathData}/pr_{phase}.txt")
    mean_lambda_ipr = np.loadtxt(f"{pathData}/mean_eigenvalues_{phase}.txt")

    if calculate_errors:
        mean_spacings_errors = np.loadtxt(f"{pathErrors}/mean_spacings_errors.txt")
        errors_Is0 = np.loadtxt(f"{pathErrors}/Is0_errors.txt")
        errors_Is0_with_kde = np.loadtxt(f"{pathErrors}/Is0_with_kde_errors.txt")
        ipr_errors = np.loadtxt(f"{pathErrors}/ipr_errors_{phase}.txt")
        pr_errors = np.loadtxt(f"{pathErrors}/pr_errors_{phase}.txt")

    else:
        mean_spacings_errors = np.zeros(len(mean_spacings))
        errors_Is0 = np.zeros(len(Is0))
        errors_Is0_with_kde = np.zeros(len(Is0_with_kde))
        ipr_errors = np.zeros(len(IPR))
        pr_errors = np.zeros(len(PR))

    # Is0 with KDE
    plt.figure()
    plt.title(r"$I_{s_0}$ with KDE", fontsize=20)
    plt.errorbar(
        mean_eigenvalues,
        Is0_with_kde,
        yerr=np.array(errors_Is0) / 2,
        fmt="x",
        barsabove=True,
        capsize=5,
        ecolor="darkorange",
        color="blue",
    )

    plt.axhline(y=0.117, color="darkorchid", linestyle="--")
    plt.axhline(y=0.398, color="plum", linestyle="--")
    plt.text(0.02, 0.398, "Poisson", verticalalignment="bottom", color="darkorchid")
    plt.text(0.005, 0.117, "RMT", verticalalignment="bottom", color="plum")
    plt.legend()
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.tight_layout()
    plt.grid(True)
    plt.ylabel(r"$I_{s0}$", fontsize=15)
    plt.show()

    # Is0 without KDE
    plt.figure()
    plt.title(r"$I_{s_0}$", fontsize=20)
    plt.errorbar(
        mean_eigenvalues,
        Is0,
        yerr=np.array(errors_Is0) / 2,
        fmt="x",
        barsabove=True,
        capsize=5,
        ecolor="darkorange",
        color="blue",
    )
    plt.axhline(y=0.117, color="darkorchid", linestyle="--")
    plt.axhline(y=0.196, color="darkorchid", linestyle="--")
    plt.axhline(y=0.398, color="plum", linestyle="--")
    plt.text(0.02, 0.398, "Poisson", verticalalignment="bottom", color="darkorchid")
    plt.text(0.005, 0.117, "RMT", verticalalignment="bottom", color="plum")
    plt.legend()
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.tight_layout()
    plt.grid(True)
    plt.ylabel(r"$I_{s0}$", fontsize=15)
    plt.show()

    # Mean Spacing
    plt.figure()
    plt.title("Mean spacings", fontsize=20)
    plt.errorbar(
        mean_eigenvalues,
        mean_spacings,
        yerr=np.array(mean_spacings_errors),
        fmt="x",
        barsabove=True,
        capsize=5,
        ecolor="darkorange",
        color="blue",
    )
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel(r"$\langle s \rangle$", fontsize=15)
    plt.axhline(y=1, color="g", linestyle="--")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # IPR
    plt.figure()
    plt.title(r" IPR vs  $\lambda$ " f" for {phase} phase", fontsize=20)
    # plt.scatter(mean_eigenvalues_dec, pr_ipr, marker="+", color="blue")
    plt.errorbar(
        mean_lambda_ipr,
        IPR,
        yerr=np.array(ipr_errors),
        fmt="x",
        barsabove=True,
        capsize=5,
        color="b",
        ecolor="g",
        label="IPR",
    )
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel(r"$\langle$ IPR $ \rangle$", fontsize=15)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # PR
    plt.figure()
    plt.title(r" PR vs  $\lambda$ " f" for {phase} phase", fontsize=20)
    plt.errorbar(
        mean_lambda_ipr,
        PR,
        yerr=np.array(pr_errors),
        fmt="x",
        barsabove=True,
        capsize=5,
        color="b",
        ecolor="g",
        label="IPR",
    )
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel(r"$\langle$ PR $ \rangle$", fontsize=15)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def lambda_edge_via_IPR():
    """reference: https://arxiv.org/pdf/1706.03562.pdf"""

    lambda_values = np.loadtxt(f"{pathData}/mean_eigenvalues_{phase}.txt")
    pr_values = np.loadtxt(f"{pathData}/ipr_{phase}.txt")
    ipr_err = np.loadtxt(f"{pathErrors}/ipr_errors_{phase}.txt")
    # ipr_values = np.loadtxt(
    #     f"{pathData}/ipr_{phase}.txt"
    # )  # Fit data with a polynomial (change degree as needed)

    coeff = np.polyfit(
        lambda_values, pr_values, deg=10
    )  # Using a 4th degree polynomial
    polynomial = np.poly1d(coeff)

    # Generate a dense grid of lambda values for plotting
    dense_lambda = np.linspace(lambda_values[0], lambda_values[-1], 1000)
    fitted_pr = polynomial(dense_lambda)

    # Compute the second derivative of the polynomial
    second_derivative = polynomial.deriv().deriv()

    # Find the roots of the second derivative to determine inflection points
    inflection_points = second_derivative.r
    valid_inflection_points = [
        point
        for point in inflection_points
        if point.imag == 0 and lambda_values[0] <= point.real <= lambda_values[-1]
    ]
    print("Inflection points (λ values):", valid_inflection_points)

    plt.figure()
    plt.title(r" IPR vs  $\lambda$ " f" for {phase} phase", fontsize=20)
    # plt.scatter(mean_eigenvalues_dec, pr_ipr, marker="+", color="blue")
    plt.errorbar(
        lambda_values,
        pr_values,
        yerr=np.array(ipr_err),
        fmt="x",
        barsabove=True,
        capsize=5,
        color="b",
        ecolor="g",
        label="IPR",
    )
    plt.plot(dense_lambda, fitted_pr, color="red", label="Fitted Polynomial Curve")
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel(r"$\langle$ IPR $ \rangle$", fontsize=15)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def lambda_edge_via_Is0():

    from scipy.optimize import curve_fit
    from scipy.interpolate import CubicSpline

    method = 1

    Is0 = np.loadtxt(f"{pathData}/Is0.txt")
    Is0_err = np.loadtxt(f"{pathErrors}/Is0_errors.txt")
    lambda_values = np.loadtxt(f"{pathData}/mean_eigenvalues.txt")

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

    if method == 2:
        spline_p = Is0 + Is0_err / 2
        spline_m = Is0 - Is0_err / 2

        cubic_spline_p = CubicSpline(lambda_values, spline_p)
        cubic_spline_m = CubicSpline(lambda_values, spline_m)

        dense_lambda = np.linspace(lambda_values[0], lambda_values[-1], 1000)
        fitted_Is0_p = cubic_spline_p(dense_lambda)
        fitted_Is0_m = cubic_spline_m(dense_lambda)

        plt.figure()
        plt.title(r"$I_{s_0}$", fontsize=20)
        plt.errorbar(
            lambda_values,
            Is0,
            yerr=np.array(Is0_err) / 2,
            fmt="x",
            barsabove=False,
            capsize=5,
            ecolor="darkorange",
            color="blue",
        )
        plt.plot(dense_lambda, fitted_Is0_p, color="red", label="Spline+")
        plt.plot(dense_lambda, fitted_Is0_m, color="blue", label="Spline-")
        plt.axhline(y=0.196, color="darkorchid", linestyle="--")

        plt.text(0.02, 0.398, "Poisson", verticalalignment="bottom", color="darkorchid")
        plt.text(0.005, 0.117, "RMT", verticalalignment="bottom", color="plum")
        plt.legend()
        plt.xlabel(r"$\lambda$", fontsize=15)
        plt.tight_layout()
        plt.grid(True)
        plt.ylabel(r"$I_{s0}$", fontsize=15)
        plt.show()


if __name__ == "__main__":
    # make_dirs()
    # spectralRegionsAnalysis()
    # topological_charge()
    IPR_and_PR()
    plotdata()
    # lambda_edge_via_IPR()
    # lambda_edge_via_Is0()
    pass
