import os, json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from dt_bim.utils.seed import set_seed
from dt_bim.utils.logging_utils import get_logger
from dt_bim.data.synth_dataset import SynthConfig, generate
from dt_bim.fusion.ekf import EKF
from dt_bim.fusion.federated import fuse_information

def main():
    logger = get_logger()
    cfg = yaml.safe_load(open('src/dt_bim/configs/default.yaml'))
    outdir = Path('data/synthetic'); outdir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.get('random_seed', 0))

    scfg = SynthConfig(
        n_tasks=cfg['n_tasks'],
        horizon=cfg['horizon'],
        state_dim=cfg['state_dim'],
        n_sensors=cfg['n_sensors'],
        dt=cfg['dt'],
        q=cfg['process_noise_q'],
        r=cfg['measurement_noise_r'],
        seed=cfg.get('random_seed', 0),
    )

    X_df, Zdfs = generate(scfg)
    X_df.to_csv(outdir/'true_states.csv', index=False)
    for k, Z in Zdfs.items():
        Z.to_csv(outdir/f'obs_{k}.csv', index=False)

    # EKF per sensor
    locals_est = []
    for k, Z in Zdfs.items():
        ekf = EKF(state_dim=scfg.state_dim, meas_dim=Z.shape[1], dt=scfg.dt,
                  q=scfg.q, r=scfg.r)
        est = []
        for _, z in Z.iterrows():
            ekf.predict()
            x, P = ekf.update(z.values)
            est.append(x.copy())
        locals_est.append((np.array(est), P))

    # Federated fusion (per time step): fuse means, approximate P by last step P
    fused = []
    for t in range(scfg.horizon):
        cur = [(locals_est[m][0][t], locals_est[m][1]) for m in range(len(locals_est))]
        xg, Pg = fuse_information(cur)
        fused.append(xg)
    fused = np.array(fused)
    fused_df = pd.DataFrame(fused, columns=[f"s{i}" for i in range(fused.shape[1])])
    fused_df.to_csv(outdir/'fused_states.csv', index=False)

    # simple "targets" for SPA: shift true vs fused
    truth = X_df.values[1:]
    pred = fused[:-1]
    spa_val = float(1.0 - np.mean(np.abs(pred - truth) / (np.abs(truth) + 1e-6)))
    json.dump({"spa_demo": spa_val}, open(outdir/'fusion_summary.json', 'w'), indent=2)
    logger.info(f"Saved fused states to {outdir/'fused_states.csv'} (SPA~{spa_val:.3f})")

if __name__ == "__main__":
    main()
