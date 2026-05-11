import os
import subprocess
import depletion_constraints as dc

print(os.getcwd())
subprocess.run(["./mf2005", "supply2.nam"], check=True)

dc.apply()
