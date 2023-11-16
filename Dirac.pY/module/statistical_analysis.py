import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as SLA
from scipy.integrate import quad
from multiprocessing import Pool
from scipy.interpolate import UnivariateSpline

potentials_list = [
    "circle",  # 0
    "square",  # 1
    "Sinai",  # 2
    "Bunimovich",  # 3
    "mod_Bunimovich",  # 4
    "mod_Sinai",  # 5
    "triangle",  # 6
    "mickey",  # 7
    "Ehrenfest",  # 8
    "Anderson",  # 9
    "Henon",  # 10
    "cardioid",  # 11
    "harm_osc",  # 12
]


potential = potentials_list[9]

plot = True
save = False
univariate = True  # if True use scipy.interpolate to unfold
unfolded = "n"

"""Select if you want the original spectrum or the unfolded one"""

if unfolded == "y":
    eig = np.loadtxt(
        f"unfolded_spectra/unfolded_spectrum_{potential}.txt", dtype=complex
    )
else:
    eig = np.loadtxt(f"spectra/eigenvalues_{potential}.txt", dtype=complex)

print(eig)
n = len(eig)
eig = eig.real
# eig = eig / SLA.norm(eig)
len_analysis = n
gamma = 0.577216
# eig = np.sort(eig)
##----------------------Staircase function working: unfolding spectra--------------------------


def unfolding_2_punto_0():
    y = np.arange(len_analysis)

    poly = np.polyfit(eig[:len_analysis], y, 50)
    poly_y = np.poly1d(poly)(eig[:len_analysis])
    plt.plot(eig[:len_analysis], poly_y, c="red", linestyle="--", label="Unfolding")
    plt.step(eig[:len_analysis], y)
    plt.xlabel("E", fontsize=15)
    plt.ylabel("N(E)", fontsize=15)
    plt.title(
        f"Staircase & Unfolding for {potential}, first {len_analysis} e.v.", fontsize=22
    )
    plt.legend()
    plt.show()

    if save:
        np.savetxt(f"unfolded_spectra/unfolded_spectrum_{potential}.txt", poly_y)


def staircase():

    lin_list = []
    n_list = []
    idx = 1
    eig1 = eig[:len_analysis]
    eig1 = np.sort(eig1)

    for e in range(len(eig1)):
        lin = 0
        lin = np.linspace(eig1[e], eig1[e - 1], 1)
        lin_list.append(lin)
        for _ in lin:
            n_list.append(idx)
        idx += 1

    lin_list = np.reshape(lin_list, len(n_list))

    return lin_list, n_list


def staircase_and_unfolding():

    """Compute the unfolding of the spectra"""

    lin_list, n_list = staircase()

    plt.figure()

    if univariate:
        poly_y = UnivariateSpline(lin_list, n_list)
        poly_y.set_smoothing_factor(
            400
        )  # per diminuire lo smoothing aumentarne il valore (dai 400 in su più o meno)

        if save:
            np.savetxt(
                f"unfolded_spectra/unfolded_spectrum_{potential}.txt", poly_y(lin_list)
            )

        plt.plot(lin_list, poly_y(lin_list), c="red", linestyle="--", label="Unfolding")

    else:
        poly = np.polyfit(lin_list, n_list, 30)
        poly_y = np.poly1d(poly)(lin_list)

        if save:
            np.savetxt(f"unfolded_spectra/unfolded_spectrum_{potential}.txt", poly_y)

        plt.plot(lin_list, poly_y, c="red", linestyle="--", label="Unfolding")

    plt.step(lin_list, n_list, c="blue", label="Staircase")
    plt.xlabel("E", fontsize=15)
    plt.ylabel("N(E)", fontsize=15)
    plt.title(
        f"Staircase & Unfolding for {potential}, first {len_analysis} e.v.", fontsize=22
    )
    plt.legend()
    plt.show()


##--------------------------------------------------------------------------

"""Da qui in poi analisi statistiche dello spettro per correlazioni a lungo raggio"""

##----------------Spectral rigidity-----------------------------------------

"""tutte le formule da pagina 110 della buonanima di Stockmann (Quantum Chaos, An Introduction)"""

##------------- Rigidità spettrale e number variance teoriche per l'ensemble GOE-------------
def spectral_rigidity_GOE(e):

    """Spectral rigidity theoretical, for N -> infty"""

    delta3 = (1 / np.pi ** 2) * (np.log(2 * np.pi * e) + gamma - 5 / 4 - np.pi ** 2 / 8)
    return delta3


def number_variance_GOE(e):

    """Number Variance theoretical for N -> infty"""

    sigma2_teorica = (2 / np.pi ** 2) * (
        np.log(2 * np.pi * e) + gamma + 1 - np.pi ** 2 / 8
    )
    return sigma2_teorica


##--------------------------------------------------------------------------------------------
##-----------Integrandi per Sigma2 e Delta3----------------------------------------


def Y2(E):
    sin = lambda x: np.sin(x) / x
    Si, _ = quad(sin, 0, E * np.pi)
    partial = (np.sin(np.pi * E) / (np.pi * E)) ** 2
    y2 = partial + (np.pi / 2 * np.sign(E) - Si) * (
        (np.cos(np.pi * E)) / (np.pi * E) - (np.sin(np.pi * E)) / (np.pi * E) ** 2
    )
    return y2


def integrand_delta3(E, L):
    integrand = (L - E) ** 3 * (2 * L ** 2 - 9 * L * E - 3 * E ** 2) * Y2(E)
    return integrand


def integrand_sigma2(E, L):
    integrand_s2 = (L - E) * Y2(E)
    return integrand_s2


##---------------------------------------------------------------------------------

##----------------------Integrazione e plot delle statistiche Delta3 e Sigma2----------------

"""Integrazione delle statistiche e confronto con plot teorico atteso dalle previsioni delle RMT"""


def delta3_integrata():

    print(f"Calcolo la rigidità spettrale per {potential}")
    delta3_list = []
    c = 0
    for e in eig[:len_analysis]:
        delta3, _ = quad(integrand_delta3, 0, e, args=(e,))
        delta3_list.append(e / 15 - (1 / (15 * e ** 4)) * delta3)
        c += 1

    L = np.linspace(0, 500, 500)
    delta3teorica = spectral_rigidity_GOE(L)

    plt.scatter(
        eig[:len_analysis], delta3_list, s=25, c="blue", label=r"$\Delta_3$ integrata"
    )
    plt.plot(L, delta3teorica, c="g", linestyle="-.", label=r"$\Delta_3$ teorica")
    plt.title(f"Rigidità spettrale {potential}", fontsize=22)
    plt.xlabel("E", fontsize=15)
    plt.ylabel(r"$\Delta_3(E)$", fontsize=15)
    plt.legend()
    plt.xlim((-50, max(eig[:len_analysis]) + 50))
    plt.show()


def sigma2_integrata():

    """Misura del Number Variance sullo spettro selezionato"""

    print(f"Calcolo number variance per {potential}")

    sigma2_list = []
    for e in eig[:len_analysis]:
        sigma2, _ = quad(integrand_sigma2, 0, e, args=(e,))
        sigma2_list.append(e - 2 * sigma2)

    L = np.linspace(min(eig[:len_analysis]), max(eig[:len_analysis]), 500)
    sigma2teorica = number_variance_GOE(L)

    plt.figure()
    plt.scatter(
        eig[:len_analysis], sigma2_list, s=25, c="red", label=r"$\Sigma_2$ integrata"
    )
    plt.plot(L, sigma2teorica, c="m", linestyle="-.", label=r"$\Sigma_2$ teorica")
    plt.title(f"Number Variance {potential}", fontsize=22)
    plt.xlabel("E", fontsize=15)
    plt.ylabel(r"$\Sigma_2 (E)$", fontsize=15)
    plt.legend()
    plt.xlim((-50, max(eig[:len_analysis]) + 50))
    plt.show()


##---------------------------------------------------------------------------------

##-----------density of states----------------------------------------------
def rho():

    """Trasformata di Fourier dello spettro"""
    eigf = eig
    ff = np.fft.fft(eigf)
    # ff = ff / SLA.norm(ff)
    t = np.arange(n)
    freq = np.fft.fftfreq(t.shape[-1])  # frequenze

    yf = np.abs((ff)[0 : len(ff) // 2]) ** 2

    if plot:
        plt.plot(freq[0 : len(ff) // 2], yf, c="green")
        plt.xlabel("f")
        plt.ylabel(r"$|\rho|^2$")
        plt.legend()
        plt.show()


def fluctuations():

    """Level energies must be unfolded"""

    eigfl = eig[:len_analysis]
    print(eigfl)
    print(eigfl)
    delta_n = []
    levelspacing = []
    for i in range(len(eigfl)):
        delta_n.append(i - eigfl[i])

    # plot fluctuations:
    plt.plot(eigfl, delta_n, label=r"$N_{fl} (E)$", c="blue")
    plt.title(f"Fluctuating part of the spectrum for {potential}", fontsize=22)
    plt.xlabel("E", fontsize=15)
    plt.ylabel(r"$N_{fl}(E)$", fontsize=15)
    plt.legend()
    plt.show()

    # verify if the spectrum is complete:
    if univariate:
        poly = UnivariateSpline(eigfl, delta_n)
        poly.set_smoothing_factor(500)
        plt.plot(eigfl, delta_n, label=r"$N_{fl} (E)$", c="blue")
        plt.plot(eigfl, poly(eigfl), label="fitting", c="green")
        plt.title(
            f"Testing the completeness of the spectrum for {potential}", fontsize=22
        )
        plt.xlabel("E", fontsize=15)
        plt.ylabel(r"$N_{fl}(E)$", fontsize=15)
        plt.legend()
        plt.show()

    else:
        poly = np.polyfit(eigfl, delta_n, 50)
        poly_delta_n = np.poly1d(poly)(delta_n)
        plt.plot(eigfl, delta_n, label=r"$N_{fl} (E)$")
        plt.step(eigfl, -poly_delta_n, label="fitting")
        plt.title(
            f"Testing the completeness of the spectrum for {potential}", fontsize=22
        )
        plt.xlabel("E", fontsize=15)
        plt.ylabel(r"$N_{fl}(E)$", fontsize=15)
        plt.legend()
        plt.show()

    # N_staircase, _ = staircase()
    # N_smooth = N_staircase + delta_n
    N_smooth = eigfl + delta_n
    # plt.plot(eigfl, range(len_analysis))
    plt.plot(N_smooth, range(len_analysis), c="violet")
    plt.ylabel(r"$N_{smooth}(E)$", fontsize=15)
    plt.xlabel("E")
    plt.title(
        r"Smooth part of the spectrum: $N_{smooth}(E)=N(E)-N_{fl}(E)$", fontsize=22
    )
    plt.show()

    # for the unfolded spectrum {eigf} the mean level spacing is 1! Steinter et al 'Mode Fluctuations as Fingerprints of Chaotic and Non-Chaotic Systems'
    for k in range(1, len(eigfl) - 1):
        levelspacing.append(eigfl[k + 1] - eigfl[k])
    print(
        "la media dello spacing è 1??", np.mean(np.array(levelspacing))
    )  # sì lo è!!!!


##---------------------------------------------------------------------------
if __name__ == "__main__":
    fluctuations()

    parallel_exe = False

    """Executing in parallel the 2 statistics"""
    from multiprocessing import Process

    if parallel_exe:
        print(f"Executing delta3 and sigma2 for {potential}")
        p1 = Process(target=delta3_integrata)
        p2 = Process(target=sigma2_integrata)
        p1.start()
        p2.start()
        p1.join()
        p2.join()

    # staircase_and_unfolding()
    # fluctuations()
    # rho()
    # unfolding_2_punto_0()
