import numpy as np
from modulo.stats import FreedmanDiaconis

s_0 = 0.508
Ns = 36
Nt = 22
Volume = Ns**3
Is0crit = 0.196
phase = "deconfined"
calculate_errors = True

if phase == "confined":
    fakevec = np.linspace(0, 1, 35600)
elif phase == "deconfined":
    fakevec = np.linspace(0, 1, 33200)

num_bins = FreedmanDiaconis(fakevec)
degrees = 3  # must be 1<= k <=5
