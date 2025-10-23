import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

def plot_covariance_samples(mean, cov, n_samples=500, ax=None, title="", color='blue'):
    """
    Plot samples from a 2D multivariate normal distribution with covariance ellipse
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Generate samples
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Plot samples
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=20, color=color, label='Samples')
    
    # Plot mean
    ax.scatter(mean[0], mean[1], c='red', s=100, marker='x', linewidths=3, label='Mean')
    
    # Compute eigenvalues and eigenvectors for ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Plot covariance ellipses (1, 2, and 3 standard deviations)
    for n_std in [1, 2, 3]:
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])
        
        ellipse = Ellipse(mean, width, height, angle=angle, 
                         fill=False, edgecolor='red', linewidth=2, 
                         linestyle='--', alpha=0.8 - n_std*0.2,
                         label=f'{n_std}σ ellipse' if n_std == 1 else '')
        ax.add_patch(ellipse)
    
    # Formatting
    ax.set_xlabel('Parameter 1', fontsize=12)
    ax.set_ylabel('Parameter 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='upper right')
    
    # Add covariance matrix as text
    cov_text = f"Cov = [[{cov[0,0]:.2f}, {cov[0,1]:.2f}]\n       [{cov[1,0]:.2f}, {cov[1,1]:.2f}]]"
    ax.text(0.02, 0.98, cov_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return ax

# Set random seed for reproducibility
np.random.seed(42)

# Mean for all distributions
mean = [0, 0]
n_samples = 500

# ============================================================================
# PART 1: Effect of Diagonal Elements (Variances)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Effect of Diagonal Elements (Variances) on Distribution', 
             fontsize=16, fontweight='bold')

# Base covariance (identity)
cov1 = np.array([[1.0, 0.0],
                 [0.0, 1.0]])
plot_covariance_samples(mean, cov1, n_samples, axes[0, 0], 
                        'Identity: Equal variances', 'blue')

# Increase variance in x-direction
cov2 = np.array([[4.0, 0.0],
                 [0.0, 1.0]])
plot_covariance_samples(mean, cov2, n_samples, axes[0, 1], 
                        'Increased σ₁² (horizontal spread)', 'green')

# Increase variance in y-direction
cov3 = np.array([[1.0, 0.0],
                 [0.0, 4.0]])
plot_covariance_samples(mean, cov3, n_samples, axes[0, 2], 
                        'Increased σ₂² (vertical spread)', 'purple')

# Decrease both variances (shrinkage)
cov4 = np.array([[0.25, 0.0],
                 [0.0, 0.25]])
plot_covariance_samples(mean, cov4, n_samples, axes[1, 0], 
                        'Shrinkage: Both variances × 0.25', 'orange')

# Different variances
cov5 = np.array([[4.0, 0.0],
                 [0.0, 0.25]])
plot_covariance_samples(mean, cov5, n_samples, axes[1, 1], 
                        'σ₁² = 4.0, σ₂² = 0.25', 'brown')

# Large variances
cov6 = np.array([[9.0, 0.0],
                 [0.0, 9.0]])
plot_covariance_samples(mean, cov6, n_samples, axes[1, 2], 
                        'Inflation: Both variances × 9', 'red')

plt.tight_layout()
plt.savefig('covariance_diagonal_effects.png', dpi=150, bbox_inches='tight')

# ============================================================================
# PART 2: Effect of Off-Diagonal Elements (Covariances/Correlations)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Effect of Off-Diagonal Elements (Covariances) on Distribution', 
             fontsize=16, fontweight='bold')

# No correlation
cov1 = np.array([[2.0, 0.0],
                 [0.0, 2.0]])
plot_covariance_samples(mean, cov1, n_samples, axes[0, 0], 
                        'No correlation (ρ = 0)', 'blue')

# Positive correlation (moderate)
cov2 = np.array([[2.0, 1.2],
                 [1.2, 2.0]])
plot_covariance_samples(mean, cov2, n_samples, axes[0, 1], 
                        'Positive correlation (ρ ≈ 0.6)', 'green')

# Strong positive correlation
cov3 = np.array([[2.0, 1.8],
                 [1.8, 2.0]])
plot_covariance_samples(mean, cov3, n_samples, axes[0, 2], 
                        'Strong positive correlation (ρ ≈ 0.9)', 'darkgreen')

# Negative correlation (moderate)
cov4 = np.array([[2.0, -1.2],
                 [-1.2, 2.0]])
plot_covariance_samples(mean, cov4, n_samples, axes[1, 0], 
                        'Negative correlation (ρ ≈ -0.6)', 'orange')

# Strong negative correlation
cov5 = np.array([[2.0, -1.8],
                 [-1.8, 2.0]])
plot_covariance_samples(mean, cov5, n_samples, axes[1, 1], 
                        'Strong negative correlation (ρ ≈ -0.9)', 'red')

# Different variances with correlation
cov6 = np.array([[4.0, 1.5],
                 [1.5, 1.0]])
plot_covariance_samples(mean, cov6, n_samples, axes[1, 2], 
                        'Different σ² with correlation', 'purple')

plt.tight_layout()
plt.savefig('covariance_offdiagonal_effects.png', dpi=150, bbox_inches='tight')

# ============================================================================
# PART 3: Progressive Changes (Animation-like sequence)
# ============================================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Progressive Covariance Changes: Shrinkage & Rotation', 
             fontsize=16, fontweight='bold')

# Progressive shrinkage
for i, scale in enumerate([1.0, 0.75, 0.5, 0.25]):
    cov = np.array([[2.0, 1.0],
                    [1.0, 2.0]]) * scale
    plot_covariance_samples(mean, cov, n_samples, axes[0, i], 
                           f'Shrinkage: scale = {scale}', 'blue')

# Progressive rotation (changing correlation)
for i, corr in enumerate([0.0, 0.5, 1.0, 1.5]):
    cov = np.array([[2.0, corr],
                    [corr, 2.0]])
    plot_covariance_samples(mean, cov, n_samples, axes[1, i], 
                           f'Covariance = {corr:.1f}', 'green')

plt.tight_layout()
plt.savefig('covariance_progressive_changes.png', dpi=150, bbox_inches='tight')

# ============================================================================
# PART 4: Statistical Summary
# ============================================================================
print("=" * 70)
print("COVARIANCE MATRIX EFFECTS SUMMARY")
print("=" * 70)
print("\n1. DIAGONAL ELEMENTS (Variances):")
print("   - Control the spread along each axis")
print("   - Larger values → wider distribution")
print("   - Smaller values → narrower distribution (convergence)")
print("   - Trace = sum of diagonal = total variance")
print("\n2. OFF-DIAGONAL ELEMENTS (Covariances):")
print("   - Control correlation between parameters")
print("   - Positive values → parameters move together (positive correlation)")
print("   - Negative values → parameters move opposite (negative correlation)")
print("   - Zero values → independent parameters")
print("   - Rotation of the ellipse depends on covariances")
print("\n3. CORRELATION COEFFICIENT:")
print("   ρ = cov(X,Y) / (σ_X * σ_Y)")
print("   - Ranges from -1 (perfect negative) to +1 (perfect positive)")
print("   - ρ = 0 means no linear relationship")
print("\n4. IN CMA-ES CONTEXT:")
print("   - Diagonal shrinking → convergence (uncertainty reduction)")
print("   - Off-diagonal learning → exploiting parameter correlations")
print("   - Condition number = λ_max / λ_min (eigenvalue ratio)")
print("   - High condition number → ellipse is stretched (anisotropic)")
print("=" * 70)

plt.show()

# ============================================================================
# BONUS: Interactive exploration function
# ============================================================================
def explore_covariance(var1=2.0, var2=2.0, cov12=0.0, n_samples=500):
    """
    Interactive function to explore covariance effects
    Usage: explore_covariance(var1=4.0, var2=1.0, cov12=1.5)
    """
    mean = [0, 0]
    cov = np.array([[var1, cov12],
                    [cov12, var2]])
    
    # Check if covariance is valid (positive semi-definite)
    eigenvalues = np.linalg.eigvalsh(cov)
    if np.any(eigenvalues < 0):
        print(f"WARNING: Invalid covariance matrix (negative eigenvalue)!")
        print(f"Eigenvalues: {eigenvalues}")
        print(f"For valid covariance: cov12² ≤ var1 × var2")
        print(f"Your constraint: {cov12}² ≤ {var1} × {var2} = {var1*var2}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_covariance_samples(mean, cov, n_samples, ax, 
                           'Custom Covariance Exploration')
    
    # Print statistics
    correlation = cov12 / (np.sqrt(var1) * np.sqrt(var2))
    print(f"\nStatistics:")
    print(f"  Variance 1 (σ₁²): {var1}")
    print(f"  Variance 2 (σ₂²): {var2}")
    print(f"  Covariance (σ₁₂): {cov12}")
    print(f"  Correlation (ρ):  {correlation:.4f}")
    print(f"  Trace:            {var1 + var2:.4f}")
    print(f"  Determinant:      {var1*var2 - cov12**2:.4f}")
    print(f"  Eigenvalues:      {eigenvalues}")
    print(f"  Condition #:      {eigenvalues.max() / eigenvalues.min():.4f}")
    
    plt.show()

print("\nTo interactively explore covariance effects, use:")
print("  explore_covariance(var1=2.0, var2=2.0, cov12=0.0)")