import numpy as np
import matplotlib.pyplot as plt

def gaussian_covariance_function(x1, x2, sigma_f=1.0, length_scale=1.0):
    """
    Gaussian (RBF) covariance function
    
    Parameters:
    x1, x2: 2D points (x, y coordinates)
    sigma_f: signal variance (controls the magnitude)
    length_scale: length scale (controls the smoothness)
    
    Returns:
    Covariance value between x1 and x2
    """
    # Calculate squared Euclidean distance
    squared_distance = np.sum((x1 - x2)**2)
    
    # Gaussian covariance function: sigma_f^2 * exp(-0.5 * d^2 / l^2)
    covariance = (sigma_f**2) * np.exp(-0.5 * squared_distance / (length_scale**2))
    
    return covariance

def compute_covariance_matrix(ensemble_points, sigma_f=1.0, length_scale=1.0):
    """
    Compute the covariance matrix for an ensemble of 2D points
    
    Parameters:
    ensemble_points: array of shape (n_points, 2) containing 2D coordinates
    sigma_f: signal variance
    length_scale: length scale
    
    Returns:
    Covariance matrix of shape (n_points, n_points)
    """
    n_points = len(ensemble_points)
    cov_matrix = np.zeros((n_points, n_points))
    
    # Fill the covariance matrix
    for i in range(n_points):
        for j in range(n_points):
            cov_matrix[i, j] = gaussian_covariance_function(
                ensemble_points[i], 
                ensemble_points[j], 
                sigma_f, 
                length_scale
            )
    
    return cov_matrix

def generate_ensemble_points(n_points=20, x_range=(-2, 2), y_range=(-2, 2)):
    """
    Generate a random ensemble of 2D points
    """
    np.random.seed(42)  # For reproducibility
    x_coords = np.random.uniform(x_range[0], x_range[1], n_points)
    y_coords = np.random.uniform(y_range[0], y_range[1], n_points)
    return np.column_stack((x_coords, y_coords))

def visualize_covariance_matrix(cov_matrix, ensemble_points, title="Covariance Matrix"):
    """
    Visualize the covariance matrix and the ensemble points
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Ensemble points
    ax1.scatter(ensemble_points[:, 0], ensemble_points[:, 1], 
               c='red', s=50, alpha=0.7)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_title('Ensemble of 2D Points')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Covariance matrix heatmap
    im = ax2.imshow(cov_matrix, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Point Index')
    ax2.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Covariance Value')
    
    plt.tight_layout()
    plt.show()

def analyze_covariance_properties(cov_matrix):
    """
    Analyze properties of the covariance matrix
    """
    print("Covariance Matrix Properties:")
    print(f"Shape: {cov_matrix.shape}")
    print(f"Determinant: {np.linalg.det(cov_matrix):.6f}")
    print(f"Trace: {np.trace(cov_matrix):.6f}")
    print(f"Condition number: {np.linalg.cond(cov_matrix):.2e}")
    print(f"Eigenvalues range: [{np.min(np.linalg.eigvals(cov_matrix)):.6f}, {np.max(np.linalg.eigvals(cov_matrix)):.6f}]")
    print(f"Is positive definite: {np.all(np.linalg.eigvals(cov_matrix) > 0)}")
    print(f"Diagonal elements (variances): {np.diag(cov_matrix)}")

# Example usage
if __name__ == "__main__":
    # Generate ensemble of 2D points
    print("Generating ensemble of 2D points...")
    ensemble = generate_ensemble_points(n_points=15)
    print(f"Generated {len(ensemble)} points")
    
    # Compute covariance matrix with different parameters
    print("\nComputing covariance matrix...")
    
    # Different parameter sets to demonstrate the effect
    parameter_sets = [
        {"sigma_f": 1.0, "length_scale": 0.5, "name": "Short length scale"},
        {"sigma_f": 1.0, "length_scale": 1.0, "name": "Medium length scale"},
        {"sigma_f": 1.0, "length_scale": 2.0, "name": "Long length scale"},
        {"sigma_f": 2.0, "length_scale": 1.0, "name": "Higher variance"}
    ]
    
    for params in parameter_sets:
        print(f"\n--- {params['name']} ---")
        cov_matrix = compute_covariance_matrix(
            ensemble, 
            sigma_f=params['sigma_f'], 
            length_scale=params['length_scale']
        )
        
        # Analyze properties
        analyze_covariance_properties(cov_matrix)
        
        # Visualize
        visualize_covariance_matrix(
            cov_matrix, 
            ensemble, 
            title=f"Covariance Matrix - {params['name']}"
        )
    
    # Demonstrate the mathematical relationship
    print("\n" + "="*50)
    print("MATHEMATICAL EXPLANATION:")
    print("="*50)
    print("The Gaussian covariance function is defined as:")
    print("k(x_i, x_j) = σ_f² * exp(-0.5 * ||x_i - x_j||² / l²)")
    print("\nWhere:")
    print("- x_i, x_j are 2D points")
    print("- ||x_i - x_j||² is the squared Euclidean distance")
    print("- σ_f is the signal variance (controls magnitude)")
    print("- l is the length scale (controls smoothness)")
    print("\nThe covariance matrix K has elements K[i,j] = k(x_i, x_j)")
    print("This matrix is symmetric (K[i,j] = K[j,i]) and positive definite")