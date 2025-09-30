import pandas as pd
import laGPy as gpr
import glob
import numpy as np

def compute_derivative_laGP(X_dv, X, y, epsilon=1e-3):
    """
    Compute derivative of laGP prediction w.r.t. X using finite differences
    """
    # Original prediction
    pred_orig = gpr.laGP(Xref=X_dv, start=10, end=15, X=X, Z=y, verb=0)
    
    # Initialize derivative array
    derivative = np.zeros_like(X_dv)
    
    # Compute partial derivatives for each dimension
    for i in range(len(X_dv)):
        X_plus = X_dv.copy()
        X_plus[0][i] += epsilon
        
        X_minus = X_dv.copy()
        X_minus[0][i] -= epsilon
        
        # Predictions at perturbed points
        pred_plus = gpr.laGP(Xref=X_plus, start=10, end=15, X=X, Z=y, verb=0)
        pred_minus = gpr.laGP(Xref=X_minus, start=10, end=15, X=X, Z=y, verb=0)
        
        # Finite difference derivative
        derivative = (pred_plus['mean'] - pred_minus['mean']) / (2 * epsilon)
    
    return derivative

# Compute the derivative


if __name__ == "__main__":
    X = pd.read_csv(glob.glob("*.0.par.csv")[0], header = 0).drop(columns=['real_name']).values
    y = pd.read_csv(glob.glob("*.0.obs.csv")[0], header = 0).drop(columns=['real_name'])
    y = y['obs'].values

    X_dv = [[0.10, -1.60]]
    pred = gpr.laGP(Xref = X_dv, start = 10, end = 15, X = X, Z = y, verb = 1)
    derivative = compute_derivative_laGP(X_dv, X, y)
    debug = 0




