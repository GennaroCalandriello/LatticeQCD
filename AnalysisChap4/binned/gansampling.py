# from GAN.GenerativeAdversarialNetwork import *
# from GAN.ProgressiveGAN import *
from GAN.TripletteGAN import *

# from GAN.const import *
import numpy as np
from main import *

num_bins = 40
NUM_EIGENVALUES = 30  # NUM_EV_FINAL
num_samples = 33000  # numero di configurazioni da generare
# START_EV = 0
# END_EV = 100


def statistics(data):
    print("data[0:10] = ", data[0:10])
    """This function calculate the spacing distribution for each bin.
    It takes bins in input structured as follows:
    bins = [config_x: [ranked_ev, real_ev], config_y: [ranked_ev, real_ev], ...]"""

    def reconstruct_triplettes(data):
        new_trip = []
        data = sorted(data, key=lambda x: x[0])

        for i in range(len(data)):
            new_trip.append([round(data[i][1]), i / len(data), data[i][0]])
        print("new_trip", new_trip[0:5])
        return new_trip

    def binning(data):

        ranked = reconstruct_triplettes(data)
        bin_size = len(ranked) // num_bins
        print("size", bin_size)
        beans = []

        # Split the spectrum in bins
        for i in range(0, len(ranked), bin_size):
            beans.append(ranked[i : i + bin_size])
        print("beans", len(beans))
        grouped_bins = []

        # Loop through all bins
        for b in beans:
            # Initialize a defaultdict to store eigenvalues of the same configuration
            grouped = defaultdict(list)

            # Loop through each [config, eigenvalue] pair in the bin
            for config, ranked_lambda, real_lambda in b:

                # Append the eigenvalue to the corresponding configuration
                grouped[config].append([ranked_lambda, real_lambda])

            # Convert the defaultdict to a regular dictionary and append it to the list
            grouped_bins.append((dict(grouped)))

        return grouped_bins

    bins = binning(data)

    mean_spacings = []
    Is0 = []
    sigma2 = []
    sigma2_errors = []
    errors_Is0 = []
    errors_mean_spacings = []
    mean_eigenvalues = []
    num = 0
    # qui ci sono tutti i bins
    for b in range(len(bins) - 1):
        print("Execution number", b)
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

        else:
            errors_Is0.append(0)
            errors_mean_spacings.append(0)

        # here I calculate Is0 and Is0 with kde
        Is0.append(compute_Is0(count_ev_in_bin, spacing))

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
        np.savetxt(f"{pathData}/sigma2_errors.txt", sigma2_errors)

    np.savetxt(f"{pathData}/mean_eigenvalues.txt", mean_eigenvalues)


def CallGAN():
    """Load the generator and the discriminator and return the generated configurations and the ranked eigenvalues"""
    from keras.models import load_model

    loaded_gen = load_model(
        "generator.h5",
        custom_objects={
            "SpectralNormalization": SpectralNormalization,
            "custom_loss": custom_loss,
        },
    )  # Carica il modello del generatore
    loaded_discr = load_model(
        "discriminator.h5",
        custom_objects={
            "custom_loss": custom_loss,
        },
    )  # Carica il modello del discriminatore

    # Genera dati casuali
    # noise = np.random.normal(0, 1, (end_idx - start_idx, num_eigenvalues))
    # from uniform distribution
    configurations = np.random.randint(0, NUM_CONFIGURATIONS, num_samples)
    # from normal distribution
    # Genera valori dalla distribuzione normale
    # media = 5
    # dev_std = 2
    # valori_flottanti = np.random.normal(loc=media, scale=dev_std, size=num_samples)
    # # Scala e arrotonda i valori per ottenere interi nel range desiderato
    # configurations = np.clip(np.round(valori_flottanti), 0, NUM_CONFIGURATIONS).astype(
    #     int
    # )

    # shuffle randomly the configurations

    configurations = configurations.reshape(-1, 1)
    positions = np.arange(0, num_samples, 1)
    # shuffle
    np.random.shuffle(positions)
    positions = positions.reshape(-1, 1)

    noise = np.random.normal(0, 1, (num_samples, NUM_EV_INITIAL))

    # generated_data = loaded_gen.predict(
    #     [noise, configurations]
    # )  # Usa il generatore per creare nuovi dati
    generated_data = loaded_gen.predict([noise, configurations, positions])
    generated_data = np.array(generated_data)
    generated_data = sorted(generated_data, key=lambda x: x[0])
    # generated_data[:][1] = generated_data[:][1] % NUM_CONFIGURATIONS
    new_data = generated_data
    # print("generated_data", generated_data[0:5])
    # for i in range(len(generated_data)):
    #     new_data.append([generated_data[i][0], int(generated_data[i][1]), i])
    # new_data = np.array(new_data)
    # print("new_data", new_data.shape)
    # print("new_data", new_data[0:5])
    statistics(np.array(new_data))


def run_model():
    # prepare the data--------------------------------------------------------------
    edeconfined, econfined, topodec, topoconf = loadtxt(topocool=True)
    edeconfined = np.abs(edeconfined[:, 4:204])
    econfined = np.abs(econfined[:, 4:204])
    econfined = econfined[:, ::2]
    edeconfined = edeconfined[:, ::2]
    edeconfined = edeconfined[:, START_EV:END_EV]

    if phase == "confined":
        data = econfined
    else:
        data = edeconfined
    # data = data.reshape(-1, NUM_EIGENVALUES)
    # -------------------------------------------------------------------------------
    train(data)

    return


def plotting():
    mean = []
    for i in range(1, num_bins - 1):
        s = np.loadtxt(f"{pathData}/spacings_{i}.txt")
        s = s / np.mean(s)
        sfake = np.linspace(min(s), max(s), len(s))
        GUE = distribution(sfake, "GUE")
        POISSON = distribution(sfake, "Poisson")

        plt.figure(figsize=(8, 7))
        plt.hist(s, bins=FreedmanDiaconis(s), histtype="step", density=True)
        plt.plot(sfake, GUE, "r--", label="GUE")
        plt.plot(sfake, POISSON, "g--", label="Poisson")
        plt.xlabel("s", fontsize=15)
        plt.ylabel("P(s)", fontsize=15)
        plt.tight_layout()
        plt.grid(True)
        plt.legend()
        plt.show()
        mean.append(np.mean(s))

    plt.figure()
    plt.scatter(range(len(mean)), mean)
    plt.show()


if __name__ == "__main__":
    # make_dirs()
    # run_model()
    CallGAN()
    plotting()
