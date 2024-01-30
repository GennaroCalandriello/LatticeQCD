import numpy as np
import matplotlib.pyplot as plt
from par import *


def jackknife_resampling(data, estimator):
    """
    Perform the Jackknife resampling on the given data.

    Parameters:
    data (array-like): The data on which to perform the resampling.
    estimator (function): The statistical estimator function to apply to the subsets of data.

    Returns:
    jackknife_estimates (array): Array of estimator values computed from each subset of the data.
    """
    n = len(data)
    jackknife_estimates = np.zeros(n)

    for i in range(n):
        # Create a subset of data excluding the i-th observation
        subset = np.delete(data, i)
        # Calculate the estimator on this subset
        jackknife_estimates[i] = estimator(subset)

    return jackknife_estimates


if __name__ == "__main__":
    # Example usage

    data = np.loadtxt(f"../data_analysis/{phase}/data/Is0.txt")
    mean_ev = np.loadtxt(f"../data_analysis/{phase}/data/mean_eigenvalues.txt")
    mean_estimator = np.mean
    jackknife_estimates = jackknife_resampling(data, mean_estimator)
    jackknife_estimates = np.sqrt((len(data) - 1) * np.var(jackknife_estimates))
    print("Jackknife Estimates:", jackknife_estimates)
    plt.errorbar(mean_ev, data, yerr=jackknife_estimates, fmt="o")
    plt.show()
