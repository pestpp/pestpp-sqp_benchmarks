import pandas as pd
import laGPy as gpr
import numpy as np
import glob


par_files = glob.glob('*[0-999].par.csv')
obs_files = glob.glob('*[0-999].obs.csv')

X_list = []
y_list = []
for par_file, obs_file in zip(par_files, obs_files):
    X_par = pd.read_csv(par_file, header=0).drop(columns=['real_name']).values
    X_list.append(X_par)
    
    y_obs = pd.read_csv(obs_file, header=0).drop(columns=['real_name']).values
    y_list.append(y_obs)

X = np.vstack(X_list)
y = np.vstack(y_list)


X_i = pd.read_csv(glob.glob('*sqp.par.csv')[0], header = 0).drop(columns=['real_name']).values #n_tr x n_dv
y = pd.read_csv(glob.glob('*sqp.obs.csv')[0], header= 0).drop(columns=['real_name']) #n_tr x 1
y = y['obs'].values


# Create and fit LaGP model
sims = gpr.laGP(
    Xref=X_dv,             
    start=6,               
    end=20,                
    X=X,                   
    Z=y,                   
    verb=10               
)
