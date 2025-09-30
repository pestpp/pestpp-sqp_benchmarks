import os
import pandas as pd
import numpy as np

def rosenbrock(x):
    x = np.array(x)
    obj = 100.0*(x[1] - x[0]**2.0)**2.0 + (1 - x[0])**2.0
    constraint = -2.25*x[0] + x[1]
    return np.array([obj, constraint])

def helper(pvals=None):
    if pvals is None:
        x = pd.read_csv("par.dat").values.reshape(-1).tolist()
    else:
        pvals_ordered = {pval: pvals[pval] for pval in sorted(pvals.index, key=lambda x: int(x[1:]))}
        x = np.array(list(pvals_ordered.values()))
    sim = {"obj": rosenbrock(x)[0], "constraint": rosenbrock(x)[1]}
    with open('obs.dat','w') as f:
        f.write('obsnme,obsval\n')
        f.write('obj,'+str(sim["obj"])+'\n')
        f.write('constraint,'+str(sim["constraint"])+'\n')
    return sim

def ppw_worker(pst_name,host,port):
    import pyemu
    ppw = pyemu.os_utils.PyPestWorker(pst_name,host,port,verbose=False)
    pvals = ppw.get_parameters()
    if pvals is None:
        return

    obs = ppw._pst.observation_data.copy()
    obs = obs.loc[ppw.obs_names,"obsval"]

    while True:

        sim = helper(pvals=pvals)

        obs.update(sim)
        
        ppw.send_observations(obs.values)
        pvals = ppw.get_parameters()
        if pvals is None:
            break


if __name__ == "__main__":
    helper()
