import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

case = '2par_one_linear_constraint'

def load_iteration_data(iteration):
    """Load parameter and observation data for a specific iteration"""
    if iteration > 0:  # For iterations > 0, we need both current and candidate data
        dv_cand_file = glob.glob(os.path.join(case, f'*{iteration}.dv_candidates.csv'))[0]
        dv_cand = pd.read_csv(dv_cand_file)
        pars_file = glob.glob(os.path.join(case, f'*{iteration-1}.par.csv'))[0]
        pars = pd.read_csv(pars_file)
    else:  # For iteration 0, we only have initial parameters
        dv_cand = None
        pars_file = glob.glob(os.path.join(case, '*0.par.csv'))[0]
        pars = pd.read_csv(pars_file)
    
    return pars, dv_cand


solution_point = np.array([1.0, 1.0])

def rosenbrock(x):
    return 100.0 * (x[1] - x[0]**2.0)**2.0 + (1 - x[0])**2.0


def constraint(x):
    return 2.25*x[0] - x[1] + 2  # This is equivalent to -2.25*x[0] + x[1] <= 2

def setup_plot():
    """Set up the base plot with contours and constraints"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate grid for contour plot
    x = np.linspace(-2.25, 2.25, 100)
    y = np.linspace(-3.0, 7.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    C = np.zeros_like(X)  # Array to store constraint values

    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = rosenbrock([X[j, i], Y[j, i]])
            C[j, i] = -2.25*X[j, i] + Y[j, i]  # Constraint values

    # Plot contours of the Rosenbrock function
    levels = np.logspace(-1, 3, 20)
    contour = ax.contourf(X, Y, Z, levels=levels, alpha=0.6, cmap='viridis')
    plt.colorbar(contour, label='Rosenbrock function value')

    # Plot contours of the constraint function
    constraint_levels = np.linspace(-4, 8, 13)
    constraint_contour = ax.contour(X, Y, C, levels=constraint_levels, colors='red', linestyles='dashed', linewidths=1)
    ax.clabel(constraint_contour, inline=True, fontsize=8, fmt='%1.1f')

    # Plot the constraint line: -2.25*x[0] + x[1] = 2
    constraint_x = np.linspace(-2.25, 2.25, 100)
    constraint_y = 2.25 * constraint_x + 2
    ax.plot(constraint_x, constraint_y, 'r-', linewidth=2, label='Constraint: -2.25*x1 + x2 = 2')

    ax.plot(solution_point[0], solution_point[1], 'bx', markersize=12, label='Solution')

    # Shade the infeasible region (where -2.25*x[0] + x[1] > 2)
    ax.fill_between(constraint_x, constraint_y, 7, alpha=0.2, color='red', label='Infeasible region')
    
    return fig, ax

def update_plot(frame, ax):
    """Update the plot for each animation frame"""
    # Clear previous points
    for artist in ax.collections + ax.lines[3:]:  # Keep contours, constraint line, and solution point
        artist.remove()
    
    # Load data for current iteration
    pars, dv_cand = load_iteration_data(frame)
    
    # Plot the current ensemble
    ax.scatter(pars['par1'], pars['par2'], c='w', marker='o', s=12, label='current ensemble')
    current_base = pars.loc[pars['real_name'] == 'base']
    ax.plot(current_base['par1'], current_base['par2'], 'go', markersize=8, 
            label=f'Base point (iter {frame}): ({current_base['par1'].values[0]:.4f}, {current_base['par2'].values[0]:.4f})')
    
    # Plot candidates if available
    if dv_cand is not None:
        ax.scatter(dv_cand['par1'], dv_cand['par2'], c='y', marker='o', s=8, label='candidates')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Constrained Rosenbrock Function - Iteration {frame}')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    return ax.collections + ax.lines[3:]  # Return the new artists


if __name__ == "__main__":
    # Set up the plot
    fig, ax = setup_plot()
    
    # Create animation
    frames = list(range(6))  # 0 to 5
    anim = FuncAnimation(fig, update_plot, frames=frames, fargs=(ax,),
                        interval=1000, blit=True)
    
    # Save animation
    writer = animation.PillowWriter(fps=1)
    anim.save('rosenbrock_optimization.gif', writer=writer)
    
    plt.close()