import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline, BSpline, splev, splrep, UnivariateSpline
from scipy.integrate import quad
from sklearn.mixture import GaussianMixture


def Sturge(data):
    length = len(data)
    return round(2 * (np.log2(length) + 1))


def PDF(spacing):

    bin_edges = np.linspace(min(spacing), max(spacing), Sturge(spacing))
    hist, _ = np.histogram(spacing, bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    bin_width = bin_edges[1] - bin_edges[0]
    spectral_density = hist / (bin_width * len(spacing))

    return spectral_density, bin_edges, bin_centers, bin_width


def CDF(data):
    """Calculate the Cumulative Density function of a set of data points,
    in this case data is the ULS"""

    a, b = 0, 0.5
    spectral_density, bin_edges, bin_centers, bin_width = PDF(data)
    par = splrep(bin_centers, spectral_density, k=3)
    x = np.linspace(bin_centers[0], bin_centers[-1], len(data))

    integral, _ = quad(b_spline, a, b, args=(par,))
    print("The CDF is: ", integral)

    return integral


def b_spline(x, par):
    return splev(x, par)


def KernelDensityFunctionIntegrator(data, num_bins, plot=False):

    hist, bin_edge = np.histogram(data, bins=num_bins)
    bin_centers = 0.5 * (bin_edge[:-1] + bin_edge[1:])

    kde = stats.gaussian_kde(data)
    xvalues = np.linspace(min(data), max(data), len(data))
    kde_values = kde(xvalues)

    if plot:
        plt.hist(
            data,
            bins=num_bins,
            density=True,
            alpha=0.5,
            histtype="step",
            label="Histogram",
        )
        plt.plot(xvalues, kde_values, label="KDE Estimator")
        plt.legend(loc="upper left")
        plt.show()

    lower_bound = 0.1
    upper_bound = 0.55
    integral_range = np.linspace(lower_bound, upper_bound, len(data))
    integral_kde = kde.integrate_box_1d(lower_bound, upper_bound)
    print("the value of integral is: ", integral_kde)

    return integral_kde

def FreedmanDiaconis(spacings, plot=False):

    q1, q3 = np.percentile(spacings, [25, 75])
    iqr = q3 - q1
    n = len(spacings)

    bin_width = 2 * iqr / (n ** (1 / 3))
    data_range = spacings.max() - spacings.min()
    num_bins = int(np.ceil(data_range / bin_width))
    Poisson = distribution(spacings, "Poisson")
    GUE = distribution(spacings, "GUE")
    GOE = distribution(spacings, "GOE")
    s = np.linspace(np.min(spacings), np.max(spacings), len(spacings))

    if plot:
        plt.hist(spacings, num_bins, histtype="step", density=True)
        plt.title("Friedman-Diaconis")
        plt.plot(s, Poisson, "g--")
        plt.plot(s, GUE, "r--", label="GUE")
        plt.plot(s, GOE, "b--", label="GOE")
        plt.legend()
        plt.show()

    return num_bins


def binning(eigenvalues, maxbins):

    bins = np.linspace(min(eigenvalues), max(eigenvalues), maxbins)
    eigenvalues = np.array(eigenvalues)
    digitized = np.digitize(eigenvalues, bins)
    binned_data = [eigenvalues[digitized == i] for i in range(1, len(bins))]

    return np.array(binned_data)

def BayesianBlocks(eigenvalues):

    from astropy.stats import bayesian_blocks
    
    """Perform Bayesian Dynamical Blocking for few statistics 
    systems. Experiment: try both with level spacing and with ordered eigenvalues"""
    bin_edges = bayesian_blocks(eigenvalues)
    s = np.linspace(min(eigenvalues), max(eigenvalues), len(eigenvalues))
    Poisson = distribution(s, "Poisson")
    GUE = distribution(s, "GUE")
    plt.hist(eigenvalues, bins=bin_edges, histtype="step", density=True, linewidth=2)
    plt.plot(s, Poisson, "g--")
    plt.plot(s, GUE, "r--")
    plt.xlabel("Unfolded Eigenvalues")
    plt.ylabel("Counts")
    plt.title("Bayesian Blocks Histogram")
    plt.show()


def GaussianMixtureModelIntegrator(data, num_bins, plot=True):

    hist, bin_edges = np.histogram(data, num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

    # perform kernel density estimator
    kde = stats.gaussian_kde(data)
    x_values = np.linspace(min(data), max(data), len(data))
    kde_values = kde(x_values)

    # fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=3)
    gmm.fit(data.reshape(-1, 1))
    gmm_pdf = np.exp(gmm.score_samples(x_values.reshape(-1, 1)))

    if plot:
        GOE = distribution(data, "GOE")
        GUE = distribution(data, "GUE")
        Poisson = distribution(data, "Poisson")
        s = np.linspace(0, max(data), len(data))
        # plot the histogram, GUE, GOE, KDE, GMM
        plt.hist(
            data,
            bins=num_bins,
            density="True",
            alpha=0.6,
            histtype="step",
            color="blue",
            label="Histogram",
        )
        plt.plot(x_values, kde_values, label="KDE")
        plt.plot(x_values, gmm_pdf, label="GMM")
        plt.plot(s, GOE, "b--", label="GOE")
        plt.plot(s, GUE, "g--", label="GUE")
        plt.plot(s, Poisson, "r--", label="Poisson")
        plt.legend(loc="upper left")
        plt.show()

    # integrate GMM PDF
    lower = 0.0
    upper = 0.51

    def gmm_integral(x):
        return np.exp(gmm.score_samples(np.array([[x]])))[0]

    integral_gmm, _ = quad(gmm_integral, lower, upper)

    return integral_gmm


def UnivariateSplineIntegrator(data, plot=False):

    n_bins = Sturge(data)
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    smoothing_factor = 5
    spline = UnivariateSpline(bin_centers, hist, s=smoothing_factor)
    x_dense = np.linspace(min(data), max(data), len(data))
    y_dense = spline(x_dense)

    if plot:

        plt.hist(
            data,
            bins=n_bins,
            density=True,
            histtype="step",
            color="blue",
            label="Histogram",
        )
        plt.plot(x_dense, y_dense, color="red", label="Spline Approx")
        plt.legend()
        plt.show()

    # integration
    lower = 0.0
    upper = 0.5
    integral_value = spline.integral(lower, upper)

    return integral_value


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
