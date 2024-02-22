import numpy as np
from modulo.stats import FreedmanDiaconis

s_0 = 0.508
Ns = 36
Nt = 22
# spectral_window =0.0018
Volume = Ns**3
Is0crit = 0.196

maxdec = 0.0518284474097687
maxconf = 0.02285429105751834
mindec = 0.0001640808829788998
minconf = 1.094816379590008e-05

# our eigenvalues
ev = 200
positive_ev = 100


phase = "deconfined"
calculate_errors = False
savedata = True
lambda_edge_with_KDE = False

if phase == "confined":
    fakevec = np.linspace(0, 1, 35600)
elif phase == "deconfined":
    fakevec = np.linspace(0, 1, 33200)

num_bins = FreedmanDiaconis(fakevec)
num_bins -= 0
degrees = 3  # must be 1<= k <=5
plotlimit = 25
