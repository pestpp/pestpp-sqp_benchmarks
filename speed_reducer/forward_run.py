import os
import math

# Read the parameters from par.dat
with open(os.path.join("par.dat"), 'r') as f:
    x = f.readline().strip().split()
    x1, x2, x3, x4, x5, x6, x7 = map(float, x)

# Objective function: Minimize weight of the speed reducer
# x1: face width (b), x2: module of teeth (m), x3: number of teeth on pinion (z)
# x4: length of first shaft between bearings (l1), x5: length of second shaft between bearings (l2)
# x6: diameter of first shaft (d1), x7: diameter of second shaft (d2)
obj = 0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934) \
    - 1.508 * x1 * (x6**2 + x7**2) \
    + 7.477 * (x6**3 + x7**3) \
    + 0.7854 * (x4 * x6**2 + x5 * x7**2)

# Constraints
g = [0] * 11

# Bending stress constraint
g[0] = 27.0 / (x1 * x2**2 * x3) - 1

# Surface stress constraint
g[1] = 397.5 / (x1 * x2**2 * x3**2) - 1

# Range of x3 constraint
g[2] = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1
g[3] = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1

# Deflection constraints
g[4] = x2 * x3 / 40.0 - 1

# Stress constraints in shafts
g[5] = x1 / 12.0 - 1
g[6] = 5.0 * x2 / x1 - 1

# Shaft stress constraints
g[7] = 1.0 / 110.0 * math.sqrt((745.0 * x4 / (x2 * x3))**2 + 16.9e6) / x6**3 - 1
g[8] = 1.0 / 85.0 * math.sqrt((745.0 * x5 / (x2 * x3))**2 + 157.5e6) / x7**3 - 1

# Dimensional constraints
g[9] = x2 * x3 / 28.0 - 1
g[10] = 5.0 - x2 * x3 / 12.0

# Write objective function value
with open(os.path.join("obs.dat"), 'w') as f:
    f.write("{0:20.8E}\n".format(obj))

# Write constraint values
with open(os.path.join("constraints.dat"), 'w') as f:
    for constraint in g:
        f.write("{0:20.8E}\n".format(constraint))