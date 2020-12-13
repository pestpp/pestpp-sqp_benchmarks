import os
import sys
import shutil
import platform
import numpy as np
import pandas as pd
import platform
import pyemu

bin_path = os.path.join("test_bin")
if "linux" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"linux")
elif "darwin" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"mac")
else:
    bin_path = os.path.join(bin_path,"win")

bin_path = os.path.abspath("test_bin")
os.environ["PATH"] += os.pathsep + bin_path


bin_path = os.path.join("..","..","..","bin")
exe = ""
if "windows" in platform.platform().lower():
    exe = ".exe"
exe_path = os.path.join(bin_path, "pestpp-sqp" + exe)


noptmax = 4
num_reals = 20
port = 4021



def basic_sqp_test():
    model_d = "mf6_freyberg"
    local=True
    if "linux" in platform.platform().lower() and "10par" in model_d:
        #print("travis_prep")
        #prep_for_travis(model_d)
        local=False
    
    t_d = os.path.join(model_d,"template")

    m_d = os.path.join(model_d,"master_sqp1")
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    pst = pyemu.Pst(os.path.join(t_d,"freyberg6_run_opt.pst"))
    pst.control_data.noptmax = 0
    pst.write(os.path.join(t_d,"freyberg6_run_sqp.pst"))
    pyemu.os_utils.run("{0} freyberg6_run_sqp.pst".format(exe_path),cwd=t_d)

    assert os.path.exists(os.path.join(t_d,"freyberg6_run_sqp.base.par"))
    assert os.path.exists(os.path.join(t_d,"freyberg6_run_sqp.base.rei"))

    pst.pestpp_options["sqp_num_reals"] = 10

    pst.control_data.noptmax = -1
    pst.write(os.path.join(t_d,"freyberg6_run_sqp.pst"))
    pyemu.os_utils.start_workers(t_d, exe_path, "freyberg6_run_sqp.pst", 
                                 num_workers=5, master_dir=m_d,worker_root=model_d,
                                 port=port)

    assert os.path.exists(os.path.join(m_d,"freyberg6_run_sqp.0.par.csv"))
    df = pd.read_csv(os.path.join(m_d,"freyberg6_run_sqp.0.par.csv"),index_col=0)
    assert df.shape == (pst.pestpp_options["sqp_num_reals"],pst.npar),str(df.shape)
    assert os.path.exists(os.path.join(m_d,"freyberg6_run_sqp.0.obs.csv"))
    df = pd.read_csv(os.path.join(m_d,"freyberg6_run_sqp.0.obs.csv"),index_col=0)
    assert df.shape == (pst.pestpp_options["sqp_num_reals"],pst.nobs),str(df.shape)



if __name__ == "__main__":
    
    shutil.copy2(os.path.join("..","exe","windows","x64","Debug","pestpp-sqp.exe"),os.path.join("..","bin","pestpp-sqp.exe"))
    basic_sqp_test()
