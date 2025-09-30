import pandas as pd
import numpy as np
import os
import glob

wdir = '2par_one_linear_constraint'
nmax = 4

pars = np.zeros((nmax+1, 2))

for i in range(nmax+1):
    with open(os.path.join(wdir, f'rosenbrock_2par_constrained_run_sqp.{i}.base.par'), 'r') as f:
        lines = f.readlines()

    par1_value = float(lines[1].strip().split()[1])
    par2_value = float(lines[2].strip().split()[1])

    pars[i, 0] = par1_value
    pars[i, 1] = par2_value

np.savetxt('pars.csv', pars, delimiter=',', fmt='%.6f')







