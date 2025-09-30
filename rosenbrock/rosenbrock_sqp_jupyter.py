import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from IPython.display import display, clear_output

# Define the Rosenbrock function
def rosenbrock(x):
    return 100.0 * (x[1] - x[0]**2.0)**2.0 + (1 - x[0])**2.0

# Define the constraint: -2.25*x[0] + x[1] <= 2
def constraint(x):
    return 2.25*x[0] - x[1] + 2  # This is equivalent to -2.25*x[0] + x[1] <= 2

# Create the base plot that will be updated by the interactive widgets
def create_plot(x1_val, x2_val):
    # Clear previous output
    clear_output(wait=True)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Generate grid for contour plot
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
    levels = np.logspace(-1, 3, 20)
    contour = plt.contourf(X, Y, Z, levels=levels, alpha=0.6, cmap='viridis')
    plt.colorbar(contour, label='Rosenbrock function value')

    # Plot contours of the constraint function
    constraint_levels = np.linspace(-4, 4, 9)
    constraint_contour = plt.contour(X, Y, C, levels=constraint_levels, colors='red', linestyles='dashed', linewidths=1)
    plt.clabel(constraint_contour, inline=True, fontsize=8, fmt='%1.1f')

    # Plot the constraint line: -2.25*x[0] + x[1] = 2
    constraint_x = np.linspace(-1.5, 1.5, 100)
    constraint_y = 2.25 * constraint_x + 2
    plt.plot(constraint_x, constraint_y, 'r-', linewidth=2, label='Constraint: -2.25*x1 + x2 = 2')

    # Shade the infeasible region (where -2.25*x[0] + x[1] > 2)
    plt.fill_between(constraint_x, constraint_y, 5, alpha=0.2, color='red', label='Infeasible region')

    # Mark the current point
    current_point = np.array([x1_val, x2_val])
    plt.plot(current_point[0], current_point[1], 'bo', markersize=10, label=f'Current point: ({x1_val:.4f}, {x2_val:.4f})')
    
    # Calculate function value and constraint status
    f_val = rosenbrock(current_point)
    constraint_val = -2.25*current_point[0] + current_point[1]
    is_feasible = constraint_val <= 2
    
    # Add text annotation with function value and feasibility
    feasibility_text = "FEASIBLE" if is_feasible else "INFEASIBLE"
    plt.annotate(f"Function value: {f_val:.4f}\nConstraint value: {constraint_val:.4f}\nStatus: {feasibility_text}",
                xy=(0.02, 0.96), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                va="top", ha="left", fontsize=10)
    
    # Add labels and legend
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Interactive Constrained Rosenbrock Function (-2.25*x1 + x2 <= 2)')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Print additional information
    print(f"Current point: x1 = {x1_val:.6f}, x2 = {x2_val:.6f}")
    print(f"Objective function value: {f_val:.6f}")
    print(f"Constraint value: {constraint_val:.6f} {'(Feasible)' if is_feasible else '(Infeasible)'}")
    
    # Show the global minimum for reference
    print("\nReference:")
    print("Unconstrained global minimum is at (1.0, 1.0) with value 0.0")

# Create interactive widgets
def interactive_rosenbrock():
    interact(
        create_plot,
        x1_val=FloatSlider(min=-1.5, max=1.5, step=0.01, value=0.0, description='x1:'),
        x2_val=FloatSlider(min=-3.0, max=5.0, step=0.01, value=0.0, description='x2:')
    )

# Run the interactive visualization
interactive_rosenbrock()