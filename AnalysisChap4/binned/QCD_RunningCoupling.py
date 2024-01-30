import numpy as np
import matplotlib.pyplot as plt

upperbeta = 6.92
lowerbeta = 5.7
Nc = 3
Nf = 3

beta_0 = 11 / 3 * Nc - 2 / 3 * Nf
beta_1 = 51 - 19 / 3 * 3


def lattice_spacing(beta):
    assert beta >= lowerbeta and beta <= upperbeta
    return 0.5 * np.exp(
        -1.6804
        - 1.7331 * (beta - 6)
        + 0.7849 * (beta - 6) ** 2
        - 0.4428 * (beta - 6) ** 3
    )


def lattice_plot():
    beta_arr = np.arange(lowerbeta, upperbeta, 0.05)
    fig, ax = plt.subplots()
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\alpha_s$")
    plt.title(r"$\alpha_s$ vs $\beta$")
    ax.plot(beta_arr, [lattice_spacing(b) for b in beta_arr])
    plt.show()


def alpha_s(mu):
    pass
