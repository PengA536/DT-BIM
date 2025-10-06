import argparse, json
import yaml, numpy as np, pandas as pd
from pathlib import Path
from tqdm import trange
from dt_bim.utils.seed import set_seed
from dt_bim.utils.logging_utils import get_logger
from dt_bim.rl.td3 import TD3Agent

def make_reward(s, a, s2, w):
    # progress: encourage norm decrease
    r_progress = -(np.linalg.norm(s2) - np.linalg.norm(s))
    # cost: small action penalty
    r_cost = -0.05 * float(np.linalg.norm(a))
    # quality: prefer states in small magnitude (proxy for good quality)
    r_quality = -0.01 * float(np.linalg.norm(s2))
    # safety: penalty on large state spikes
    r_safety = -0.05 * float(np.maximum(0.0, np.linalg.norm(s2) - 3.0))
    r = w["progress"]*r_progress + w["cost"]*r_cost + w["quality"]*r_quality + w["safety"]*r_safety
    return float(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logger = get_logger()
    cfg = yaml.safe_load(open('src/dt_bim/configs/default.yaml'))
    set_seed(args.seed)
    outdir = Path('data/synthetic'); outdir.mkdir(parents=True, exist_ok=True)

    fused = pd.read_csv(outdir/'fused_states.csv').values
    state_dim = fused.shape[1]
    action_dim = cfg["action_dim"]
    td3_cfg = cfg["td3"]
    if args.steps is not None:
        td3_cfg["steps"] = args.steps

    agent = TD3Agent(state_dim, action_dim, td3_cfg)
    w = cfg["reward_weights"]

    s = fused[0]
    log = []
    for t in trange(td3_cfg["steps"], desc="TD3"):
        a = agent.act(s, noise=True)
        # simple dynamics proxy for RL: s2 = s + small controlled drift + noise
        s2 = s + 0.05*a + np.random.normal(0, 0.02, size=s.shape)
        r = make_reward(s, a, s2, w)
        d = 0.0
        agent.buffer.push(s, a, r, s2, d)
        info = agent.update() or {}
        s = s2
        if (t+1) % 100 == 0:
            log.append({"step": t+1, "reward": r, **{k: float(v) for k,v in info.items()}})

    import pandas as pd
    pd.DataFrame(log).to_csv(outdir/"rl_train_log.csv", index=False)
    # Save a last greedy action as guidance for MIP
    a_star = agent.act(s, noise=False)
    np.save(outdir/"rl_guidance.npy", a_star)
    summary = {"final_step": td3_cfg["steps"], "last_reward": float(log[-1]["reward"]) if log else 0.0}
    json.dump(summary, open(outdir/"rl_summary.json", "w"), indent=2)
    logger.info("TD3 training finished. Guidance vector saved.")

if __name__ == "__main__":
    main()
