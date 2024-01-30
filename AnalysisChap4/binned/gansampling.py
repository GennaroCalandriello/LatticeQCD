from GAN.GenerativeAdversarialNetwork import *

# from GAN.const import *
import numpy as np
from main import *

num_bins = 17


def generate_eigenvalues(generator, num_samples):
    noise = np.random.normal(0, 1, size=[num_samples, NUM_EIGENVALUES])
    generated_eigenvalues = generator.predict(noise)
    return generated_eigenvalues


def spectralRegions():
    """Qui seleziono un certo range e calcolo la differenza s(i,j)= x(i,j+1) - x(i,j)
    reference: Localization properties of Dirac modes at the Roberge-Weiss phase transition, PHYS. REV. D 105, 014506 (2022)
    """

    ranked_dec, _, _ = load_and_save()
    ranked_ = ranked_dec
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

    statistics(grouped_bins)


def statistics(bins):
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
    num = 0
    # qui ci sono tutti i bins
    for b in range(len(bins) - 1):
        num += 1
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
            f"{pathData}/spacings_{num}.txt",
            spacing,
        )
        # -------------------------------------------------------------------------------------

        # here I calculate the error on Is0 and on the mean spacing
        # kind= 1 for Is0, kind=2 for mean spacing
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


def load_and_save():
    from keras.models import load_model

    num_samples = 400

    loaded_gen = load_model(
        "path/to/generator.h5",
        custom_objects={"SpectralNormalization": SpectralNormalization},
    )  # Carica il modello del generatore
    loaded_discr = load_model(
        "path/to/discriminator.h5"
    )  # Carica il modello del discriminatore
    noise = np.random.normal(
        0, 1, size=[num_samples, NUM_EIGENVALUES]
    )  # Genera rumore casuale
    generated_data = loaded_gen.predict(
        noise
    )  # Usa il generatore per creare nuovi dati

    configurations_dec = len(generated_data[:, 0])

    unsorted_dec = []
    ranked_dec = []
    ranked_conf = []

    # 1. ordino tutti gli autovalori lambda(i,j) di tutte le configurazioni
    for i in range(len(generated_data[:, 0])):
        for j in range(len(generated_data[0, :])):
            unsorted_dec.append([int(i + 1), generated_data[i, j]])

    sorted_dec = sorted(unsorted_dec, key=lambda x: x[1])

    # rimpiazzo i lambda(i,j) con il loro rango r nell'insieme di tutti gli autovalori diviso
    # il numero di configurazioni
    rank1 = np.arange(1, len(sorted_dec[:]) + 1, 1)
    rank1 = rank1 / configurations_dec

    # here in the ranked_dec I append: [configurazione, rango, lambda(i,j)]
    for k in range(len(rank1)):
        ranked_dec.append([sorted_dec[k][0], rank1[k], sorted_dec[k][1]])

    return ranked_dec, ranked_conf, configurations_dec


def run_model():
    edeconfined, econfined, topodec, topoconf = loadtxt(topocool=True)
    print(topodec.shape)
    edeconfined = np.abs(edeconfined[:, 4:204])
    econfined = np.abs(econfined[:, 4:204])
    econfined = econfined[:, ::2]
    edeconfined = edeconfined[:, ::2]
    edeconfined = edeconfined[:, START_EV:END_EV]

    if phase == "confined":
        data = econfined
    else:
        data = edeconfined
    data = data.reshape(-1, NUM_EIGENVALUES)
    build_and_train(data)

    return


if __name__ == "__main__":
    # make_dirs()
    # run_model()
    spectralRegions()
    mean = []
    # distribuzione

    for i in range(17):
        s = np.loadtxt(f"{pathData}/spacings_{i+1}.txt")
        sfake = np.linspace(min(s), max(s), len(s))
        GUE = distribution(sfake, "GUE")
        POISSON = distribution(sfake, "Poisson")

        plt.figure()
        plt.hist(s, bins=FreedmanDiaconis(s), histtype="step", density=True)
        plt.plot(sfake, GUE, "r--", label="GUE")
        plt.plot(sfake, POISSON, "g--", label="Poisson")
        plt.show()
        mean.append(np.mean(s))
    plt.figure()
    plt.scatter(range(len(mean)), mean)
    plt.show()

    pass
