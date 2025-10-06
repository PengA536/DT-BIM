#!/usr/bin/env bash
set -e
python scripts/simulate_and_fuse.py
python scripts/train_td3.py --steps 5000 --seed 0
python scripts/run_mip.py --seed 0
