import pyemu
import sys

def run():
    sys.path.insert(0, "template")

    from forward_run import ppw_worker as ppw_function

    pyemu.os_utils.start_workers("template","pestpp-mou","ros_2par_constr.pst",
                                num_workers=8,
                                master_dir="master",worker_root='.',
                                verbose=True,
                                ppw_function=ppw_function)
    sys.path.remove("template")

if __name__ == "__main__":
    run()
