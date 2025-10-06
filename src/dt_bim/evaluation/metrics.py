import numpy as np

def spa(pred, truth, eps=1e-6):
    pred = np.asarray(pred); truth = np.asarray(truth)
    return float(1.0 - np.mean(np.abs(pred - truth) / (np.abs(truth) + eps)))

def rue(usage, availability):
    usage = np.asarray(usage); availability = np.asarray(availability)
    den = availability.sum() + 1e-9
    return float((usage * availability).sum() / den)

def csr(baseline_cost, optimized_cost):
    return float((baseline_cost - optimized_cost) / (baseline_cost + 1e-9))

def qcr(q, qmin=0.85):
    q = np.asarray(q)
    return float((q >= qmin).mean())
