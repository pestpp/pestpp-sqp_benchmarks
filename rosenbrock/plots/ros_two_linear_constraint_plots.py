import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import re
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation

NMAX = 3
case = '2par_two_linear_constraint_infeas'

c1 = -0.5
c2 = 0.0 

def rosenbrock(x):
    return 100.0 * (x[1] - x[0]**2.0)**2.0 + (1 - x[0])**2.0

def constraint1(x):
    return -2.25*x[0] + x[1] - c1

def constraint2(x):
    return x[0] + 1.5*x[1] - c2

def plot_rosenbrock(iter=1, ax=None):
    pattern = os.path.join("..", case, '*.par.csv')
    all_par_files = glob.glob(pattern)
    pars_file = None
    for file in all_par_files:
        basename = os.path.basename(file)
        match = re.search(r'(\d+)\.par\.csv$', basename)
        if match and int(match.group(1)) == iter:
            pars_file = file
            break
    
    if pars_file is None:
        raise FileNotFoundError(f"Could not find par.csv file for iteration {iter}")
    
    pars = pd.read_csv(pars_file).drop(columns=['real_name'])

    pattern = os.path.join("..", case, '*.base.par')
    all_base_files = glob.glob(pattern)
    base_par_file = None
    for file in all_base_files:
        basename = os.path.basename(file)
        match = re.search(r'(\d+)\.base\.par$', basename)
        if match and int(match.group(1)) == iter:
            base_par_file = file
            break
    
    if base_par_file is None:
        raise FileNotFoundError(f"Could not find base.par file for iteration {iter}")
    
    with open(base_par_file, 'r') as f:
        lines = f.readlines()
        par1 = float([line for line in lines if 'par1' in line][0].split()[1])
        par2 = float([line for line in lines if 'par2' in line][0].split()[1])

    # Create meshgrid for contours
    x = np.linspace(-2.2, 2.2, 100)
    y = np.linspace(-2.2, 2.2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    C1 = np.zeros_like(X)  # Array to store first constraint values
    C2 = np.zeros_like(X)  # Array to store second constraint values

    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = rosenbrock([X[j, i], Y[j, i]])
            C1[j, i] = -2.25*X[j, i] + Y[j, i]  # First constraint values
            C2[j, i] = X[j, i] + 1.5*Y[j, i]    # Second constraint values

    # Plot contours of the Rosenbrock function
    levels = np.logspace(-1, 3, 25)
    contour = ax.contourf(X, Y, Z, levels=levels, alpha=0.6, cmap='viridis')
    
    # Plot contours of the first constraint function
    constraint_levels = np.linspace(-4, 4, 9)
    constraint_contour1 = ax.contour(X, Y, C1, levels=constraint_levels, colors='red', linestyles='dashed', linewidths=1)
    ax.clabel(constraint_contour1, inline=True, fontsize=8, fmt='%1.1f')
    
    # Plot contours of the second constraint function
    constraint_contour2 = ax.contour(X, Y, C2, levels=constraint_levels, colors='indigo', linestyles='dashed', linewidths=1)
    ax.clabel(constraint_contour2, inline=True, fontsize=8, fmt='%1.1f')

    # Plot the first constraint line: -2.25*x[0] + x[1] = c1
    constraint_x = np.linspace(-2.2, 2.2, 100)
    constraint1_y = 2.25 * constraint_x + c1
    ax.plot(constraint_x, constraint1_y, 'r-', linewidth=2, label='Constraint 1: -2.25*x1 + x2 = ' + str(c1))

    # Plot the second constraint line: x[0] + 1.5*x[1] = c2
    constraint2_y = (-constraint_x - c2) / 1.5
    ax.plot(constraint_x, constraint2_y, ls='-', color='indigo', linewidth=2, label='Constraint 2: x1 + 1.5*x2 = ' + str(c2))

    # Shade the infeasible regions
    ax.fill_between(constraint_x, constraint1_y, 5, alpha=0.2, color='red', label='Infeasible region 1')
    ax.fill_between(constraint_x, constraint2_y, 5, alpha=0.2, color='indigo', label='Infeasible region 2')

    # BASE point
    current_point = np.array([par1, par2])
    ax.scatter(current_point[0], current_point[1], c='b', marker='o', s=30, zorder=20, label=f'base: ({par1:.4f}, {par2:.4f}), {rosenbrock(current_point):.4f}')
    # Solution
    solution_point = np.array([0.17143, -0.11429])
    ax.plot(solution_point[0], solution_point[1], 'rx', markersize=10, label='Solution')
    # Ensemble members
    pars = pars.values
    ax.scatter(pars[:, 0], pars[:, 1], c='w', marker='o', s=15, zorder=10, label='')
    # Candidate points
    if (iter < NMAX):
        pattern = os.path.join("..", case, f'*{iter+1}.dv_candidates.csv')
        matching_files = glob.glob(pattern)
        dv_cand_file = None
        for file in matching_files:
            basename = os.path.basename(file)
            match = re.search(r'(\d+)\.dv_candidates\.csv$', basename)
            if match and int(match.group(1)) == iter + 1:
                dv_cand_file = file
                break
        
        if dv_cand_file is None:
            raise FileNotFoundError(f"Could not find dv_candidates file for iteration {iter + 1}")
        
        dv_cand = pd.read_csv(dv_cand_file).drop(columns=['real_name'])
        cands = dv_cand.values
        ax.scatter(cands[:, 0], cands[:, 1], ec='g', c = 'none', marker='o', s=10, zorder=10)    # Add labels and legend
 
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim((-2.25, 2.25))
    ax.set_ylim((-2.25, 2.25))
    ax.set_title('Constrained Rosenbrock Function with Two Linear Constraints\n Ensemble PESTPP-SQP - Iteration = ' + str(iter))
    ax.legend(bbox_to_anchor=(-.05, -0.08), loc='upper left', borderaxespad=0.)
    ax.grid(True)

    return ax

def create_advanced_animation(nmax = 1):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    plt.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.9)
    
    def animate(frame):
        ax.clear()
        plot_rosenbrock(iter=frame, ax=ax)
        
        return ax,
    
    anim = FuncAnimation(fig, animate, frames=nmax+1, interval=1000, blit=False, repeat=True)
    
    # Save as MP4 (requires ffmpeg)
    try:
        anim.save('rosenbrock_animation.mp4', writer='ffmpeg', fps=1/3, bitrate=1800, savefig_kwargs={'bbox_inches': 'tight'})
        print("MP4 video saved as 'rosenbrock_animation.mp4'")
    except Exception as e:
        print(f"MP4 creation failed: {e}")
        print("Saving as GIF instead...")
        anim.save('rosenbrock_animation.gif', writer='pillow', fps=1.25, savefig_kwargs={'bbox_inches': 'tight'})
        print("GIF saved as 'rosenbrock_animation.gif'")
    
    plt.show()
    return anim

def create_individual_plots(nmax = 1):
    """Create individual static plots for each iteration"""
    for iter in range(0, nmax+1):
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_rosenbrock(iter=iter, ax=ax)
        # plt.savefig(f'iteration_{iter}.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":

    create_individual_plots(nmax = NMAX)
    
    # Create the advanced animation
    # create_advanced_animation(nmax = NMAX)