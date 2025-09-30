import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation

NMAX = 6
case = '2par_one_linear_constraint'
c = -1.5

def rosenbrock(x):
    return 100.0 * (x[1] - x[0]**2.0)**2.0 + (1 - x[0])**2.0

def constraint(x):
    return 2.25*x[0] - x[1] + c  # This is equivalent to -2.25*x[0] + x[1] <= 2

def plot_rosenbrock(iter=1, ax=None):
    # Load data
    pars_file = glob.glob(os.path.join("..",case, f'*{iter}.par.csv'))[0]
    pars = pd.read_csv(pars_file).drop(columns=['real_name'])

    with open(glob.glob(os.path.join("..",case, f'*{iter}.base.par'))[0], 'r') as f:
        lines = f.readlines()
        par1 = float([line for line in lines if 'par1' in line][0].split()[1])
        par2 = float([line for line in lines if 'par2' in line][0].split()[1])

    # Create meshgrid for contours
    x = np.linspace(-2.2, 2.2, 100)
    y = np.linspace(-2.2, 2.2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    C = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = rosenbrock([X[j, i], Y[j, i]])
            C[j, i] = -2.25*X[j, i] + Y[j, i]

    # Plot contours of the Rosenbrock function
    levels = np.logspace(-1, 3, 25)
    contour = ax.contourf(X, Y, Z, levels=levels, alpha=0.6, cmap='viridis')
    
    # Add colorbar only once (outside the animation function)
    # if not hasattr(plot_rosenbrock, 'colorbar_added'):
    # cbar = plt.colorbar(contour, ax=ax, label='Rosenbrock function value')
    # plot_rosenbrock.colorbar_added = True

    # Plot contours of the constraint function
    constraint_levels = np.linspace(-4, 4, 9)
    constraint_contour = ax.contour(X, Y, C, levels=constraint_levels, colors='red', linestyles='dashed', linewidths=1)
    ax.clabel(constraint_contour, inline=True, fontsize=8, fmt='%1.1f')

    # Plot the constraint line: -2.25*x[0] + x[1] = c
    constraint_x = np.linspace(-2.2, 2.2, 100)
    constraint_y = 2.25 * constraint_x + c
    ax.plot(constraint_x, constraint_y, 'r-', linewidth=2, label='Constraint: -2.25*x1 + x2 = ' + str(c))

    # Shade the infeasible region (where -2.25*x[0] + x[1] > 2)
    ax.fill_between(constraint_x, constraint_y, 5, alpha=0.2, color='red', label='Infeasible region')

    # BASE point
    current_point = np.array([par1, par2])
    ax.scatter(current_point[0], current_point[1], c='b', marker='o', s=30, zorder=20, label=f'base: ({par1:.4f}, {par2:.4f}), {rosenbrock(current_point):.4f}')
    # Solution
    solution_point = np.array([1.1224, 1.0254])
    ax.plot(solution_point[0], solution_point[1], 'rx', markersize=10, label='Solution')
    # Ensemble members
    pars = pars.values
    ax.scatter(pars[:, 0], pars[:, 1], c='w', marker='o', s=15, zorder=10, label='')
    # Candidate points
    if (iter < NMAX):
        dv_cand_file = glob.glob(os.path.join("..",case, f'*{iter+1}.dv_candidates.csv'))[0]
        dv_cand = pd.read_csv(dv_cand_file).drop(columns=['real_name'])
        cands = dv_cand.values
        ax.scatter(cands[:, 0], cands[:, 1], ec='g', c = 'none', marker='o', s=10, zorder=10)

    # Add labels and legend
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim((-2.25, 2.25))
    ax.set_ylim((-2.25, 2.25))
    ax.set_title('Constrained Rosenbrock Function (-2.25*x1 + x2 <= ' + str(c) + ')\n Ensemble PESTPP-SQP - Iteration = ' + str(iter))
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
    # Uncomment the line below if you want individual plots first
    create_individual_plots(nmax = NMAX)
    
    # Create the advanced animation
    # create_advanced_animation(nmax = NMAX)