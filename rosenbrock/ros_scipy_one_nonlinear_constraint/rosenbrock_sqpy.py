import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the Rosenbrock function
def rosenbrock(x):
    return 100.0 * (x[1] - x[0]**2.0)**2.0 + (1 - x[0])**2.0

# Define the constraint: -2.25*x[0] + x[1] <= 2
# For inequality constraints in scipy.optimize.minimize, the constraint is in the form g(x) >= 0
# So we need to negate the original constraint to make it: 0 <= -(-2.25*x[0] + x[1] - 2) = 2.25*x[0] - x[1] + 2
def constraint(x):
    return 2.25*x[0]**2 - x[1] - 1.5  # This is equivalent to -2.25*x[0] + x[1] <= 2

# Set up the constraint dictionary for scipy.optimize.minimize
cons = ({'type': 'ineq', 'fun': constraint})

# Initial guess
x0 = np.array([0.10, -0.60])  # Changed initial guess to be in the feasible region

# Create a list to store intermediate solutions
iterations = [x0.copy()]

# Callback function to record intermediate solutions
def callback(xk):
    iterations.append(xk.copy())
    return False

# Solve using SQP
result = minimize(rosenbrock, x0, method='SLSQP', constraints=cons,
                  callback=callback, options={'disp': True, 'ftol': 1e-9})

print("\nOptimization Results:")
print(f"Success: {result.success}")
print(f"Status: {result.message}")
print(f"Iterations: {result.nit}")
print(f"Function evaluations: {result.nfev}")
print(f"Optimal point: x1 = {result.x[0]:.6f}, x2 = {result.x[1]:.6f}")
print(f"Objective function value: {result.fun:.6f}")
print(f"Constraint value at solution: {-2.25*result.x[0] + result.x[1]:.6f}")  # Original constraint form
print(f"Constraint function value: {constraint(result.x):.6f}")  # Should be >= 0 at solution
