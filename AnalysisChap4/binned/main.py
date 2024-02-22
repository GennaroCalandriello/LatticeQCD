import numpy as np
import multiprocessing
from functools import partial
from collections import defaultdict
from scipy.optimize import curve_fit


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

    print("maxdeconfined", maxdec)
    print("maxconfined", maxconf)

    return econfined, edeconfined, maxdec, mindec, maxconf, minconf


def loadSortRank():
    edeconfined, econfined, topodec, topoconf = loadtxt(topocool=True)

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
    sigma2 = []
    sigma2_errors = []
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
                spacing.append(added_s)

            # else:
            #     added_s = 0
            # # -------------------------------------------------------------------------------------------------------------------------------------

            for e in range(len(ranked_ev) - 1):
                spacing.append(((ranked_ev[e + 1][0] - ranked_ev[e][0])))

            for e in range(len(ranked_ev)):
                count_ev_in_bin += 1
                eigenvalueList.append(ranked_ev[e][1])

            # spacing.append(added_s)

        # spacing for each bin in which the spectrum is divided
        spacing = np.array(spacing)
        # s2temp = compute_sigma2(spacing)
        # s2temp = s2temp - np.mean(spacing) ** 2
        s2temp = np.mean(spacing**2)  # - np.var(spacing)
        sigma2.append(s2temp)

        eigenvalueList = np.array(eigenvalueList)

        # SAVE spacings for histogram plot-------------------------------------------------------
        np.savetxt(
            f"{pathData}/spacings_{round(min(eigenvalueList), 4)}-{round(max(eigenvalueList), 4)}.txt",
            spacing,
        )
        # -------------------------------------------------------------------------------------

        # here I calculate the error on Is0 and on the mean spacing
        # kind= 1 for Is0, kind=2 for mean spacing
        print(spacing)
        if calculate_errors:
            sigma2_errors.append(errorAnalysis(count_ev_in_bin, spacing, kind=5))
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
    sigma2 = np.array(sigma2)
    np.savetxt(f"{pathData}/Is0.txt", Is0)
    np.savetxt(f"{pathData}/sigma2.txt", sigma2)

    np.savetxt(f"{pathData}/mean_spacings.txt", mean_spacings)

    if calculate_errors:
        np.savetxt(f"{pathErrors}/mean_spacings_errors.txt", errors_mean_spacings)
        np.savetxt(f"{pathErrors}/Is0_errors.txt", errors_Is0)
        np.savetxt(f"{pathErrors}/Is0_with_kde_errors.txt", errors_Is0_with_kde)
        np.savetxt(f"{pathData}/sigma2_errors.txt", sigma2_errors)

    np.savetxt(f"{pathData}/mean_eigenvalues.txt", mean_eigenvalues)
    np.savetxt(f"{pathData}/Is0_with_kde.txt", Is0_with_kde)


def spectralRegionsAnalysis():
    """Qui seleziono un certo range e calcolo la differenza s(i,j)= x(i,j+1) - x(i,j)
    reference: Localization properties of Dirac modes at the Roberge-Weiss phase transition, PHYS. REV. D 105, 014506 (2022)
    """

    print("Freedman-Diaconis number of bins: ", num_bins)

    # here in the ranked_dec I append: [configurazione, rango, lambda(i,j)]
    ranked_dec, ranked_conf, configurations_dec, configurations_conf = loadSortRank()

    if phase == "deconfined":
        ranked_ = ranked_dec
        configurations = configurations_dec
    elif phase == "confined":
        ranked_ = ranked_conf
        configurations = configurations_conf

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
    print(d[0, 22 * ev + 200 : 22 * ev + 204 + 22])

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
        ipr_sum = np.zeros((configurations, positive_ev))
        for e in range(positive_ev):
            for c in range(configurations):
                ipr_sum[c][e] = sum(new_ipr2[c][e][:])
        ipr_sum = np.array(ipr_sum)

        for e in range(positive_ev):
            ipr_errors.append(errorAnalysis(None, ipr_sum[:, e], kind=3))
        # for e in range(positive_ev):
        #     print("calculate errors for eigenvalues #:", e)
        #     errors_temp = []

        #     # for c in range(configurations):
        #     #     for t in range(Nt):
        #     #         errors_temp.append(sum(new_ipr2[c][e][t]))
        #     # errors_temp = np.array(errors_temp)
        #     # ipr_errors.append(errorAnalysis(None, errors_temp, kind=3))

        # ipr_errors = np.array(ipr_errors)

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
    only_ipr = True

    pathDataconf = "data_analysis/confined/data/"
    pathErrorsconf = "data_analysis/confined/errors/"
    Is0_with_kde = np.loadtxt(f"{pathData}/Is0_with_kde.txt")
    Is0 = np.loadtxt(f"{pathData}/Is0.txt")
    mean_eigenvalues = np.loadtxt(f"{pathData}/mean_eigenvalues.txt")
    mean_spacings = np.loadtxt(f"{pathData}/mean_spacings.txt")
    IPR = np.loadtxt(f"{pathData}/ipr_{phase}.txt")
    # PR = np.loadtxt(f"{pathData}/pr_{phase}.txt") / Volume
    mean_lambda_ipr = np.loadtxt(f"{pathData}/mean_eigenvalues_{phase}.txt")
    mean_lambda_ipr_conf = np.loadtxt(f"{pathDataconf}/mean_eigenvalues_confined.txt")
    IPRconf = np.loadtxt(f"{pathDataconf}/ipr_confined.txt")
    print("LUHGOOOOOOOOO", len(IPRconf))
    # PRconf = np.loadtxt(f"{pathDataconf}/pr_confined.txt")

    if calculate_errors:
        mean_spacings_errors = np.loadtxt(f"{pathErrors}/mean_spacings_errors.txt")
        errors_Is0 = np.loadtxt(f"{pathErrors}/Is0_errors.txt")
        errors_Is0_with_kde = np.loadtxt(f"{pathErrors}/Is0_with_kde_errors.txt")
        ipr_errors = np.loadtxt(f"{pathErrors}/ipr_errors_{phase}.txt")
        ipr_errors_confined = np.loadtxt(f"{pathErrorsconf}/ipr_errors_confined.txt")
        pr, pr_errors = PR(IPR, ipr_errors)
        PRconf, pr_errors_confined = PR(IPRconf, ipr_errors_confined)
        # pr_errors = np.zeros(len(IPR))
        # pr_errors_confined = np.zeros(len(IPRconf))

    else:
        mean_spacings_errors = np.zeros(len(mean_spacings))
        errors_Is0 = np.zeros(len(Is0))
        errors_Is0_with_kde = np.zeros(len(Is0_with_kde))
        ipr_errors = np.zeros(len(IPR))
        pr_errors = np.zeros(len(IPR))
        ipr_errors_confined = np.zeros(len(IPRconf))
        pr_errors_confined = np.zeros(len(IPRconf))

    # Is0 with KDE
    # ipr_errors *= np.sqrt(len(IPR))
    if not only_ipr:
        plt.figure(figsize=(8, 7))
        plt.title(r"$I_{s_0}$, KDE, " f"{phase} phase", fontsize=16)

        plt.errorbar(
            mean_eigenvalues[0:plotlimit],
            Is0_with_kde[0:plotlimit],
            yerr=np.array(errors_Is0_with_kde[0:plotlimit]) / 2,
            fmt="x",
            barsabove=True,
            capsize=5,
            color="r",
            ecolor="b",
            label=r"$I_{s_0}$",
            elinewidth=0.8,
        )
        plt.axhline(y=0.117, color="darkorchid", linestyle="--", linewidth=1.2)
        plt.axhline(y=0.398, color="plum", linestyle="--", linewidth=1.2)
        plt.axhline(y=0.196, color="red", linestyle="-.", linewidth=1.2)
        plt.text(
            0.02,
            0.398,
            "Poisson",
            verticalalignment="bottom",
            color="darkorchid",
            fontsize=14,
        )
        plt.text(
            0.005, 0.117, "RMT", verticalalignment="bottom", color="plum", fontsize=14
        )
        plt.legend(loc="upper right", fontsize=14)
        plt.xlabel(r"$\lambda$", fontsize=15)
        plt.tight_layout()
        plt.ylim(0, 0.45)
        plt.grid(True)
        plt.ylabel(r"$I_{s0}$", fontsize=15)
        plt.show()

        # Is0 without KDE

        plt.figure(figsize=(8, 7))
        plt.title(r"$I_{s_0}$" f"{phase} phase", fontsize=16)
        # Add shaded region
        errors_Is0 = np.array(errors_Is0)
        # errors_Is0 = errors_Is0 / np.sqrt(len(Is0))
        errors_Is0 = errors_Is0 / 2
        plt.errorbar(
            mean_eigenvalues[0:plotlimit],
            Is0[0:plotlimit],
            yerr=np.array(errors_Is0[0:plotlimit]) / 2,
            fmt="x",
            barsabove=True,
            capsize=5,
            color="blue",
            ecolor="orangered",
            label=r"$I_{s_0}$",
            elinewidth=0.8,
        )

        plt.axhline(y=0.117, color="darkorchid", linestyle="--", linewidth=1.2)
        plt.axhline(y=0.196, color="red", linestyle="-.", linewidth=1.2)
        plt.axhline(y=0.398, color="plum", linestyle="--", linewidth=1.2)

        if phase == "confined":
            plt.text(
                0.002,
                0.398,
                "Poisson",
                verticalalignment="bottom",
                color="darkorchid",
                fontsize=14,
            )
            plt.text(
                -0.0004,
                0.117,
                "RMT",
                verticalalignment="bottom",
                color="plum",
                fontsize=14,
            )
        if phase == "deconfined":
            plt.text(
                0.02,
                0.398,
                "Poisson",
                verticalalignment="bottom",
                color="darkorchid",
                fontsize=14,
            )
            plt.text(
                0.005,
                0.117,
                "RMT",
                verticalalignment="bottom",
                color="plum",
                fontsize=14,
            )

        plt.legend(loc="upper right", fontsize=14)
        plt.xlabel(r"$\lambda$", fontsize=15)
        plt.tight_layout()
        plt.grid(True)
        plt.ylabel(r"$I_{s_{0}}$", fontsize=15)
        plt.show()

        # Mean Spacing

        plt.figure(figsize=(8, 7))
        plt.title(f"Mean spacings {phase} phase", fontsize=16)
        plt.errorbar(
            mean_eigenvalues[0:plotlimit],
            mean_spacings[0:plotlimit],
            yerr=np.array(mean_spacings_errors[0:plotlimit]) / 2,
            fmt="x",
            barsabove=True,
            capsize=5,
            color="plum",
            ecolor="darkorchid",
            label="Mean spacings",
            elinewidth=0.8,
        )
        plt.xlabel(r"$\lambda$", fontsize=15)
        plt.ylabel(r"$\langle s \rangle$", fontsize=15)
        plt.axhline(y=1, color="g", linestyle="--")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    # IPR
    plt.figure(figsize=(8, 7))
    plt.title(r" IPR vs  $\lambda$ " f" for both phases", fontsize=16)

    plt.errorbar(
        mean_lambda_ipr,
        IPR,
        yerr=np.array(ipr_errors) / 2,
        fmt="x",
        barsabove=True,
        capsize=5,
        color="green",
        ecolor="blue",
        elinewidth=0.8,
        label="IPR deconfined",
    )
    plt.errorbar(
        mean_lambda_ipr_conf,
        IPRconf,
        yerr=np.array(ipr_errors_confined) * 3,
        fmt="x",
        barsabove=True,
        capsize=5,
        color="red",
        ecolor="orange",
        elinewidth=0.8,
        label="IPR confined",
    )
    plt.axvspan(
        0.01490,
        0.01558,
        color="blue",
        alpha=0.1,
        linestyle="dashed",
        linewidth=0.5,
        label=r"$\lambda_{c}$",
    )

    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel(r"$\langle$ IPR $ \rangle$", fontsize=15)
    plt.tight_layout()
    plt.legend(loc="upper right", fontsize=14)
    plt.grid(True)
    plt.show()

    # PR
    _, pr_errors = PR(np.loadtxt(f"{pathErrors}/pr_errors_deconfined.txt"), ipr_errors)
    plt.figure(figsize=(8, 7))
    plt.title(r" PR vs  $\lambda$ " f" for both phases", fontsize=16)
    plt.errorbar(
        mean_lambda_ipr,
        PR,
        yerr=np.array(pr_errors),
        fmt="x",
        barsabove=True,
        capsize=5,
        color="b",
        ecolor="g",
        label="PR deconfined",
        elinewidth=0.8,
    )
    plt.errorbar(
        mean_lambda_ipr_conf,
        PRconf / Volume,
        yerr=np.array(pr_errors_confined),
        fmt="x",
        barsabove=True,
        capsize=5,
        color="r",
        ecolor="orange",
        label="PR confined",
        elinewidth=0.8,
    )
    plt.axvspan(
        0.01490,
        0.01558,
        color="blue",
        alpha=0.1,
        linestyle="dashed",
        linewidth=0.5,
        label=r"$\lambda_{c}$",
    )
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel(r"$\langle$ PR $ \rangle$", fontsize=15)
    plt.tight_layout()
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True)
    plt.show()


def lambda_edge_via_IPR():
    """reference: https://arxiv.org/pdf/1706.03562.pdf"""

    lambda_values = np.loadtxt(f"{pathData}/mean_eigenvalues_{phase}.txt")
    pr_values = np.loadtxt(f"{pathData}/pr_{phase}.txt")
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
    plt.title(r" IPR vs  $\lambda$ " f" for {phase} phase", fontsize=16)
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
        plt.title(r"$I_{s_0}$", fontsize=16)
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


def cumulative_fill_fraction():
    PR = np.loadtxt(f"{pathData}/pr_{phase}.txt")
    PRerrors = np.loadtxt(f"{pathErrors}/pr_errors_{phase}.txt")
    PR /= Volume
    PR /= max(PR)
    PRup = PR + PRerrors
    PRdown = PR - PRerrors
    lambda_values = np.loadtxt(f"{pathData}/mean_eigenvalues_{phase}.txt")
    CVFF = np.zeros_like(PR)
    CVFFup = np.zeros_like(PR)
    CVFFdown = np.zeros_like(PR)

    i = 0
    for lamb in lambda_values:
        indices = np.where(lambda_values < lamb)[0]
        cvff = PR[indices].sum()
        cvffup = PRup[indices].sum()
        cvffdown = PRdown[indices].sum()
        CVFF[i] = cvff
        CVFFup[i] = cvffup
        CVFFdown[i] = cvffdown
        i += 1

    # Plot the cumulative fill fraction as a function of lambda
    plt.figure(figsize=(8, 7))
    # plt.plot(lambda_values, CVFF, label="CVFF", linestyle="--", color = "blue", linewidth=0.8)
    plt.fill_between(lambda_values, CVFFdown, CVFFup, alpha=0.3, color="red")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("CVFF")
    plt.title(r"Cumulative Volume Fill Fraction vs $\lambda$")
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.show()


def spectral_density():
    # Load the data
    from scipy.stats import gaussian_kde
    from scipy.integrate import simps  # For numerical integration

    econfined, edeconfined, _, _, _, _ = loading()
    ranked_dec, ranked_conf, configurations_dec, configurations_conf = loadSortRank()

    # reshape the data from 332, 100 to 33200, 1
    if phase == "deconfined":
        eigenvalues = edeconfined.flatten()
    else:
        eigenvalues = econfined.flatten()
    # eigenvalues = ranked_dec[:][:][1]
    # eigenvalues = np.array(eigenvalues)
    # eigenvalues = eigenvalues.flatten()

    eigenvalues = np.sort(eigenvalues, axis=0)
    # Calculate the kernel density estimate
    kde = gaussian_kde(eigenvalues)

    # Create a range of values over which to evaluate the KDE
    min_eigenvalue, max_eigenvalue = eigenvalues.min(), eigenvalues.max()
    x_d = np.linspace(min_eigenvalue, max_eigenvalue, 1000)

    # Evaluate the KDE
    kde_values = kde(x_d)

    # Normalize the KDE values
    # Ensure the total area under the KDE curve is 1
    kde_area = simps(kde_values, x_d)
    kde_values_normalized = kde_values / kde_area

    # Plot the spectral density
    plt.figure(figsize=(8, 7))
    plt.plot(x_d, kde_values_normalized, label="KDE", color="blue")

    # Color under the curve up to lambda = 0.015
    lambda_fill = 0.01524
    mask = x_d <= lambda_fill
    plt.fill_between(x_d[mask], kde_values_normalized[mask], color="red", alpha=0.5)

    # Annotations and labels
    plt.xlabel("λ")
    plt.ylabel("ρ(λ)")
    plt.title(f"Spectral density of near-zero modes, {phase} phase")
    plt.legend()
    plt.grid(True)
    plt.show()


def extract_lambda_interval(filename):
    # Remove 'spacings_' prefix and '.txt' suffix
    stripped_name = filename[len("spacings_") : -len(".txt")]

    # Split the string at the dash '-'
    interval_parts = stripped_name.split("-")

    # Convert the parts to floats
    lambda_start = float(interval_parts[0])
    lambda_end = float(interval_parts[1])

    return lambda_start, lambda_end


def calculate_errors_Hist(counts, bin_edges, total_count):
    """
    Calculate the errors for histogram bins when the histogram is normalized.
    :param counts: Array of counts in each bin of the histogram.
    :param bin_edges: Array of bin edge values.
    :param total_count: Total number of samples in the histogram.
    :return: Array of errors for each bin.
    """
    bin_widths = np.diff(bin_edges)
    errors = np.sqrt(counts * total_count * bin_widths) / total_count
    return errors / bin_widths


def plot_histograms():
    import re

    # Assuming pathData is defined and holds the correct path to your data
    for file in os.listdir(pathData):
        if file.startswith("spacings"):
            spacings = np.loadtxt(f"{pathData}/{file}")

            # find the range of lambda in the name of file
            range_lambda = extract_lambda_interval(file)

            sfake = np.linspace(min(spacings), max(spacings), len(spacings))
            GUE = distribution(sfake, "GUE")
            POISSON = distribution(sfake, "Poisson")
            print("range lambda", range_lambda)

            # Use FreedmanDiaconis to determine bin width
            bin_count = FreedmanDiaconis(spacings)

            # Calculate histogram data
            counts, bin_edges = np.histogram(spacings, bins=bin_count, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Calculate errors
            errors = calculate_errors_Hist(counts, bin_edges, len(spacings))

            plt.figure(figsize=(8, 7))
            # plt.title(f"Histogram of spacings", fontsize=16)

            # Plot histogram as step
            plt.hist(
                spacings,
                bins=bin_count,
                density=True,
                histtype="step",
                color="goldenrod",
                label=f"λ = {range_lambda[0]}-{range_lambda[1]}",
            )

            # Add error bars with horizontal bars between errors

            plt.errorbar(
                bin_centers,
                counts,
                yerr=errors,
                fmt=".",
                color="goldenrod",
                ecolor="darkorchid",
                barsabove=True,
                elinewidth=0.6,
                capsize=3,
            )

            # Plot GUE and Poisson distributions
            plt.plot(sfake, GUE, "r--", label="GUE")
            plt.plot(sfake, POISSON, "g--", label="Poisson")

            plt.xlabel("s", fontsize=15)
            plt.ylabel("p(s)", fontsize=15)
            plt.grid(True)
            plt.xlim(0, 5)
            plt.tight_layout()
            plt.legend()
            plt.show()


def spatial_extension_ev():
    IPR = np.loadtxt(f"{pathData}/ipr_{phase}.txt")
    eigen = np.loadtxt(f"{pathData}/mean_eigenvalues_{phase}.txt")
    IPR_err = np.loadtxt(f"{pathErrors}/ipr_errors_{phase}.txt")
    S_psi = 1 / (Nt * IPR) ** (1 / 3)
    # Error propagations
    S_psi_err = 1 / 3 * (Nt * IPR) ** (-4 / 3) * IPR_err * Nt
    plt.figure(figsize=(8, 7))
    plt.title(f"Spatial extension of eigenmodes, {phase} phase", fontsize=16)
    plt.errorbar(
        eigen,
        S_psi,
        yerr=S_psi_err,
        fmt="x",
        color="blue",
        ecolor="red",
        capsize=5,
        barsabove=True,
        label="S",
        elinewidth=0.8,
    )
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel(r"$S_\psi$", fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def sigma2_Is0():
    sigma2 = np.loadtxt(f"{pathData}/sigma2.txt")
    s2errors = np.loadtxt(f"{pathData}/sigma2_errors.txt")
    mean_eigenvalues_dec = np.loadtxt(f"{pathData}/mean_eigenvalues.txt")
    # remove the last 4 points from sigma2
    sigma2 = sigma2[1:-5]
    lam_array = mean_eigenvalues_dec[1:-5]
    s2errors = s2errors[1:-5]

    def fit_function(lam, A, B, C):
        # Ensure numerical stability by limiting the argument of tanh
        safe_B = np.clip(B, -2, 2)
        argument = safe_B * (lam - C)
        tanh_term = np.tanh(argument)
        return A * (1 - np.tanh(B * (lam - 0.35))) + (3 / (8 * np.pi)) - 1

    # Using error bars in the fitting process
    # Initial guesses for A, B, and C
    initial_guesses = [0, 0, 2]

    # Bounds for A, B, and C (if applicable)
    lower_bounds = [-10, -1, -1]  # Example lower bounds
    upper_bounds = [200, 20, 20]  # Example upper bounds

    # Fit the function to the data with bounds and initial guesses
    popt, pcov = curve_fit(
        fit_function,
        lam_array,
        sigma2,
        p0=initial_guesses,
        bounds=(lower_bounds, upper_bounds),
        maxfev=100000,
    )

    # Extracting the best fit values
    A_fit, B_fit, C_fit = popt
    print("A =", A_fit)
    print("B =", B_fit)
    print("C =", C_fit)

    # Generate fitted sigma^2 values using the fit_function and the optimized parameters
    fitted_sigma2 = fit_function(lam_array, *popt)

    # Plotting the original data and the fitted curve
    plt.errorbar(
        lam_array,
        sigma2,
        yerr=s2errors,
        fmt="x",
        color="blue",
        label="Data",
        capsize=5,
        barsabove=True,
        elinewidth=0.8,
    )
    # plt.plot(
    #     lam_array,
    #     fitted_sigma2,
    #     color="red",
    #     label="Fitted Curve",
    #     linewidth=0.8,
    #     linestyle="--",
    # )
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel(r"$\langle s^2 \rangle$", fontsize=15)
    # add horizontali line at y = 1.17
    plt.axhline(y=1.18, color="darkorchid", linestyle="--", label="GUE", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    plt.show()


def plot_dirac_spectrum():
    ranked_dec, ranked_conf, _, _ = loadSortRank()
    edeconfined, econfined, topodec, topoconf = loadtxt(topocool=True)
    # create a list with all the eigenvalues in the first 10 configurations

    plt.figure(figsize=(8, 7))
    plt.title("Dirac Spectrum")
    labels = ["x", "^", "s", "v"]
    for i in range(4):
        plt.plot(
            edeconfined[i][4:204],
            labels[i],
            label=f"Configuration {i+1}",
            markersize=2,
        )
    plt.legend(fontsize=12)
    plt.xlabel("n", fontsize=15)
    plt.ylabel(r"$\lambda_n$", fontsize=15)
    # add horizontal line at y =0
    plt.axhline(y=0, color="blue", linestyle="-.", linewidth=0.8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def IPR_and_PR_duepuntozero():
    # load the data (remember that IPR^(-1) \approx Ns^3*PR)
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
    mean_eigenvalues_ = mean_eigenvalues(
        np.abs(d[:, 4:204])
    )  # here the mean on all configurations of all \lambda
    # ipr_ = d[:, 204 : 22 * ev + 204]
    d = d[:, 204:]
    ipr_ = np.zeros((configurations, Nt * positive_ev))
    print("shape of ipr_", ipr_.shape)

    for i in range(configurations):
        selected_elements = []

        for j in range(0, Nt * ev, 2 * Nt):
            selected_elements.extend(d[i, j : j + Nt])

        print("len selected elements", len(selected_elements))
        ipr_[i] = np.array(selected_elements)

    ipr_errors = np.loadtxt(f"{pathErrors}/ipr_errors_{phase}.txt")
    ipr = np.zeros((configurations, positive_ev))

    # sum ipr over all time slices for each \lam for each configuration
    alcuni_ipr_t = []
    for i in range(configurations):
        for j in range(positive_ev):
            ipr[i][j] = np.mean(ipr_[i][j * Nt : (j + 1) * Nt])
            if i == 3:
                alcuni_ipr_t.append(ipr_[i][j * Nt : (j + 1) * Nt])

    ipr_mean = np.zeros((positive_ev))
    for i in range(positive_ev):
        ipr_mean[i] = np.mean(ipr[:, i])

    pr, pr_errors = PR(ipr_mean, ipr_errors)

    # SAVE DATA
    np.savetxt(f"{pathData}/ipr_{phase}.txt", ipr_mean)
    np.savetxt(f"{pathData}/pr_{phase}.txt", pr)
    np.savetxt(f"{pathData}/mean_eigenvalues_{phase}.txt", mean_eigenvalues_)
    np.savetxt(f"data_analysis/{phase}/ipr_t_{phase}.txt", alcuni_ipr_t)


def PR(ipr, ipr_errors):

    pr = 1 / (ipr * Ns**3 * Nt)
    pr_errors = ipr_errors / (ipr * Ns**3 * Nt) ** 2

    return pr, pr_errors


def errorFunction():
    "calculate the errors for the IPR and PR and save them in a file"
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
    mean_eigenvalues_ = mean_eigenvalues(
        np.abs(d[:, 4:204])
    )  # here the mean on all configurations of all \lambda
    # ipr_ = d[:, 204 : 22 * ev + 204]
    d = d[:, 204:]
    ipr_ = np.zeros((configurations, Nt * positive_ev))
    print("shape of ipr_", ipr_.shape)

    for i in range(configurations):
        selected_elements = []

        for j in range(0, Nt * ev, 2 * Nt):
            selected_elements.extend(d[i, j : j + Nt])

        print("len selected elements", len(selected_elements))
        ipr_[i] = np.array(selected_elements)

    new_ipr = []
    ipr_errors = (np.loadtxt(f"{pathErrors}/ipr_errors_{phase}.txt")).tolist()
    ipr = np.zeros((configurations, positive_ev))

    ipr_mean = np.zeros((positive_ev))
    for i in range(positive_ev):
        ipr_mean[i] = np.mean(ipr[:, i])

    pr, pr_errors = PR(ipr_mean, ipr_errors)

    # sum ipr over all time slices for each \lam for each configuration

    for e in range(positive_ev):
        err = errorAnalysis(None, ipr[:, e], kind=3)
        ipr_errors.append(err)
    print("last error", err)
    np.savetxt(f"{pathErrors}/ipr_errors_{phase}.txt", ipr_errors)
    np.savetxt(f"{pathErrors}/pr_errors_{phase}.txt", pr_errors)
    np.savetxt(f"{pathErrors}/pr_errors_{phase}.txt", pr_errors)


def some_time_slices_for_IPR():

    from scipy.stats import gaussian_kde

    alcuni_ipr = np.loadtxt(f"data_analysis/{phase}/ipr_t_{phase}.txt")
    # perform a kerne density estimation
    plt.figure(figsize=(8, 7))
    for i in range(100):
        if i % 10 == 0:
            kde = gaussian_kde(alcuni_ipr[i])
            x_d = np.linspace(min(alcuni_ipr[i]), max(alcuni_ipr[i]), 1000)
            kde_values = kde(x_d)
            plt.plot(x_d, kde_values)
    plt.title("IPR for some time slices, deconfined")
    plt.xlabel("IPR", fontsize=15)
    plt.ylabel("t", fontsize=15)
    plt.show()

    alcuni_ipr_confinati = np.loadtxt(f"data_analysis/confined/ipr_t_confined.txt")
    # perform a kerne density estimation
    plt.figure(figsize=(8, 7))
    for i in range(100):
        if i % 10 == 0:
            kde = gaussian_kde(alcuni_ipr_confinati[i])
            x_d = np.linspace(
                min(alcuni_ipr_confinati[i]), max(alcuni_ipr_confinati[i]), 1000
            )
            kde_values = kde(x_d)
            plt.plot(x_d, kde_values)
    plt.title("IPR for some time slices, confined")
    plt.xlabel("IPR", fontsize=15)
    plt.ylabel("t", fontsize=15)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def ParticipationPlot():

    mean_lambda_ipr = np.loadtxt(
        f"data_analysis/deconfined/data/mean_eigenvalues_deconfined.txt"
    )
    mean_lambda_ipr_conf = np.loadtxt(
        f"data_analysis/confined/data/mean_eigenvalues_confined.txt"
    )

    # Dceonfined
    IPR = np.loadtxt(f"data_analysis/deconfined/data/ipr_deconfined.txt")
    ipr_errors = np.loadtxt(
        f"data_analysis/deconfined/errors/ipr_errors_deconfined.txt"
    )
    PRatio, pr_errors = PR(IPR, ipr_errors)

    # confined
    IPRconf = np.loadtxt(f"data_analysis/confined/data/ipr_confined.txt")
    ipr_errors_confined = np.loadtxt(
        f"data_analysis/confined/errors/ipr_errors_confined.txt"
    )
    PRconf, pr_errors_confined = PR(IPRconf, ipr_errors_confined)

    # IPR
    plt.figure(figsize=(8, 7))
    plt.title(r" IPR vs  $\lambda$ " f" for both phases", fontsize=16)

    plt.errorbar(
        mean_lambda_ipr,
        IPR,
        yerr=np.array(ipr_errors) / 2,
        fmt="x",
        barsabove=True,
        capsize=5,
        color="green",
        ecolor="blue",
        elinewidth=0.8,
        label="IPR deconfined",
    )
    plt.errorbar(
        mean_lambda_ipr_conf,
        IPRconf,
        yerr=np.array(ipr_errors_confined) * 3,
        fmt="x",
        barsabove=True,
        capsize=5,
        color="red",
        ecolor="orange",
        elinewidth=0.8,
        label="IPR confined",
    )
    plt.axvspan(
        0.01490,
        0.01558,
        color="blue",
        alpha=0.1,
        linestyle="dashed",
        linewidth=0.5,
        label=r"$\lambda_{c}$",
    )

    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel(r"$\langle$ IPR $ \rangle$", fontsize=15)
    plt.tight_layout()
    plt.legend(loc="upper right", fontsize=14)
    plt.grid(True)
    plt.show()

    # PR
    plt.figure(figsize=(8, 7))
    plt.title(r" PR vs  $\lambda$ " f" for both phases", fontsize=16)
    plt.errorbar(
        mean_lambda_ipr,
        PRatio,
        yerr=np.array(pr_errors),
        fmt="x",
        barsabove=True,
        capsize=5,
        color="b",
        ecolor="g",
        label="PR deconfined",
        elinewidth=0.8,
    )
    plt.errorbar(
        mean_lambda_ipr_conf,
        PRconf / Volume,
        yerr=np.array(pr_errors_confined),
        fmt="x",
        barsabove=True,
        capsize=5,
        color="r",
        ecolor="orange",
        label="PR confined",
        elinewidth=0.8,
    )
    plt.axvspan(
        0.01490,
        0.01558,
        color="blue",
        alpha=0.1,
        linestyle="dashed",
        linewidth=0.5,
        label=r"$\lambda_{c}$",
    )
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel(r"$\langle$ PR $ \rangle$", fontsize=15)
    plt.tight_layout()
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("PHASE: ___ ", phase)
    # Here, the function you can call
    #
    # make_dirs()
    # spectralRegionsAnalysis()

    # plot_histograms()
    # topological_charge()
    errorFunction()
    # IPR_and_PR_duepuntozero()
    # some_time_slices_for_IPR()

    # plotdata()
    ParticipationPlot()
    # sigma2_Is0()

    # lambda_edge_via_IPR()
    # lambda_edge_via_Is0()
    # spectral_density()
    # spatial_extension_ev()
    # cumulative_fill_fraction()
    # IPR_and_PR()
    # plot_dirac_spectrum()
