import os
with open(os.path.join("par.dat"),'r') as f:
	x = f.readline().strip().split()
	x1, x2, x3 = float(x[0]), float(x[1]), float(x[2])

result = 100.0*(x2 - x1**2.0)**2.0 + (1 - x1)**2.0 + x3
# see: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.rosen.html

constraint = -2.25*x1**2 + x2 


with open(os.path.join("obs.dat"),'w') as f:
	f.write("{0:20.8E}\n".format(result))

with open(os.path.join("constraints.dat"),'w') as f:
	f.write("{0:20.8E}\n".format(constraint))