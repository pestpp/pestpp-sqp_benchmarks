import pyemu

# Read existing PEST control file
pst = pyemu.Pst("rosenbrock_2par_constrained_run_sqp.pst")

# Get parameter names and bounds
par_names = pst.parameter_names
par_bounds = pst.parameter_data[['parval1', 'parlbnd', 'parubnd']]

# Create covariance based on parameter bounds
# Use 10% of parameter range as standard deviation
std_devs = []
for _, row in par_bounds.iterrows():
    par_range = row['parubnd'] - row['parlbnd']
    std_devs.append(0.1 * par_range)

# Create diagonal covariance matrix
cov_matrix = pyemu.Cov.from_parameter_data(par_names, std_devs)

# Save to file
cov_matrix.to_ascii("prior_cov.mat")