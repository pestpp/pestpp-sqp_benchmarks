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
    return 2.25*x[0] - x[1] + 2  # This is equivalent to -2.25*x[0] + x[1] <= 2

# Set up the constraint dictionary for scipy.optimize.minimize
cons = ({'type': 'ineq', 'fun': constraint})

# Initial guess
x0 = np.array([-1.3333, -1.0])  # Changed initial guess to be in the feasible region

# Create a list to store intermediate solutions
iterations = [x0.copy()]

# Callback function to record intermediate solutions
def callback(xk):
    iterations.append(xk.copy())
    return False

# Solve using SQP
result = minimize(rosenbrock, x0, method='SLSQP', constraints=cons,
                  options={'disp': True, 'ftol': 1e-9})

print("\nOptimization Results:")
print(f"Success: {result.success}")
print(f"Status: {result.message}")
print(f"Iterations: {result.nit}")
print(f"Function evaluations: {result.nfev}")
print(f"Optimal point: x1 = {result.x[0]:.6f}, x2 = {result.x[1]:.6f}")
print(f"Objective function value: {result.fun:.6f}")
print(f"Constraint value at solution: {-2.25*result.x[0] + result.x[1]:.6f}")  # Original constraint form
print(f"Constraint function value: {constraint(result.x):.6f}")  # Should be >= 0 at solution

# Visualize the solution
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-3.0, 5.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
C = np.zeros_like(X)  # Array to store constraint values

for i in range(len(x)):
    for j in range(len(y)):
        Z[j, i] = rosenbrock([X[j, i], Y[j, i]])
        C[j, i] = -2.25*X[j, i] + Y[j, i]  # Constraint values

# Plot contours of the Rosenbrock function
plt.figure(figsize=(10, 8))
levels = np.logspace(-1, 3, 20)
contour = plt.contourf(X, Y, Z, levels=levels, alpha=0.6, cmap='viridis')
plt.colorbar(contour, label='Rosenbrock function value')

# Plot contours of the constraint function
constraint_levels = np.linspace(-4, 4, 9)  # Constraint contour levels
constraint_contour = plt.contour(X, Y, C, levels=constraint_levels, colors='red', linestyles='dashed', linewidths=1)
plt.clabel(constraint_contour, inline=True, fontsize=8, fmt='%1.1f')

# Plot the constraint line: -2.25*x[0] + x[1] = 2
constraint_x = np.linspace(-1.5, 1.5, 100)
constraint_y = 2.25 * constraint_x + 2
plt.plot(constraint_x, constraint_y, 'r-', linewidth=2, label='Constraint: -2.25*x1 + x2 = 2')

# Shade the infeasible region (where -2.25*x[0] + x[1] > 2)
plt.fill_between(constraint_x, constraint_y, 3, alpha=0.2, color='red', label='Infeasible region')

# Convert iterations list to numpy array for easier plotting
iterations_array = np.array(iterations)

# Plot the path of the optimization
# plt.plot(iterations_array[:, 0], iterations_array[:, 1], 'b-o', markersize=4, alpha=0.7, label='Optimization path')

# Mark the initial point
plt.plot(iterations_array[0, 0], iterations_array[0, 1], 'go', markersize=8, label='Initial point')

# Mark the optimal point
plt.plot(result.x[0], result.x[1], 'ro', markersize=10, label=f'Optimal: ({result.x[0]:.4f}, {result.x[1]:.4f})')

# Add labels and legend
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Constrained Rosenbrock Optimization with SQP (-2.25*x1 + x2 <= 2)')
plt.legend()
plt.grid(True)
plt.savefig('rosenbrock_sqp_solution_leq.png', dpi=300)
plt.show()

# Save results to files in the same format as the original script
with open("par.dat", 'w') as f:
    f.write(f"{result.x[0]} {result.x[1]}")

with open("obs.dat", 'w') as f:
    f.write("{0:20.8E}\n".format(rosenbrock(result.x)))

with open("constraints.dat", 'w') as f:
    # Using the original constraint form for consistency with the problem statement
    f.write("{0:20.8E}\n".format(-2.25*result.x[0] + result.x[1]))