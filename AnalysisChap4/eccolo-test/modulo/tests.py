import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def SpectralDensity(data, rng, N_conf):

    low = rng[0]
    high = rng[1]
    data = np.abs(data)
    data = data[:, 4:204]
    data = np.array(data[:, ::2])

    # through N_conf you can choose a single configuration or more
    # keep ev from data:
    ev = []
    for i in range(N_conf):
        for j in range(len(data[0, :])):
            if data[i, j] >= low and data[i, j] <= high:
                ev.append(data[i, j])

    kde = gaussian_kde(ev)
    eig_range = np.linspace(np.min(ev), np.max(ev), len(ev))

    rho = kde(eig_range)

    plt.plot(eig_range, rho)
    plt.show()


def staircase_function(eigenvalues):

    sorted_ev = np.sort(eigenvalues)
    N = len(sorted_ev)

    return np.array(
        [(i, val) for i, val in enumerate(sorted_ev, start=1)], dtype=np.float64
    )


def ensemble_average_staircase(staircases):
    return np.mean(staircases, axis=0)


def unfold_eigenvalues(eigenvalues, ensemble_staircase):
    unfolded_ev = []

    for val in eigenvalues:
        idx = np.searchsorted(ensemble_staircase[:, 1], val)
        unfolded_value = (
            ensemble_staircase[idx, 0]
            if idx < len(ensemble_staircase)
            else ensemble_staircase[-1, 0]
        )
        unfolded_ev.append(unfolded_value)

    return np.array(unfolded_ev)

def unfoldingprova(eig):

    len_analysis = len(eig)
    y = np.arange(len_analysis)
    # eig = np.sort(eig)

    poly = np.polyfit(eig[:len_analysis], y, 10)
    poly_y = np.poly1d(poly)(eig[:len_analysis])

    return poly_y

if __name__=='__main__':
    ###STEPS:

    # Load eigenvalues from different configurations
    configurations = [np.random.rand(50) for _ in range(10)]

    #compute staircase for each confs
    staircases = [staircase_function(eigenvalues) for eigenvalues in configurations]
    #compute the ensemble average for each configuration
    ensemble_staircase = ensemble_average_staircase(staircases)
    #perform unfolding
    unfolded_configurations = [unfold_eigenvalues(eigenvalues, ensemble_staircase) for eigenvalues in configurations]

