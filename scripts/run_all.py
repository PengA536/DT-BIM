from simulate_and_fuse import main as sim_main
from train_td3 import main as rl_main
from run_mip import main as mip_main

if __name__ == "__main__":
    sim_main()
    rl_main()
    mip_main()
