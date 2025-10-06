import numpy as np
import pandas as pd
import pulp

def solve_small_assignment(n_tasks=20, n_resources=6, guide=None, seed=0):
    rs = np.random.RandomState(seed)
    cost = rs.uniform(1.0, 5.0, size=(n_tasks, n_resources))
    duration = rs.uniform(1.0, 4.0, size=n_tasks)

    # Optional "guide" from RL action embedding to prefer some pairs
    if guide is None:
        guide = np.zeros((n_tasks, n_resources))
    guide = (guide - guide.min()) / (guide.max() - guide.min() + 1e-6)
    cost = cost - 0.5 * guide  # encourage guided pairs

    prob = pulp.LpProblem("ResourceAssign", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (range(n_tasks), range(n_resources)), lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Each task gets exactly one resource
    for i in range(n_tasks):
        prob += pulp.lpSum([x[i][j] for j in range(n_resources)]) == 1

    # Resource capacity: at most ceil(n_tasks/n_resources) tasks per resource (toy)
    cap = int(np.ceil(n_tasks / n_resources))
    for j in range(n_resources):
        prob += pulp.lpSum([x[i][j] for i in range(n_tasks)]) <= cap

    # Objective: cost + simple lateness proxy (sum of durations per resource overload)
    obj = pulp.lpSum(cost[i, j] * x[i][j] for i in range(n_tasks) for j in range(n_resources))
    prob += obj

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assign = np.zeros((n_tasks, n_resources), dtype=int)
    for i in range(n_tasks):
        for j in range(n_resources):
            assign[i, j] = int(pulp.value(x[i][j]) > 0.5)

    # Build schedule start times (greedy serial per resource)
    starts = np.zeros(n_tasks)
    for j in range(n_resources):
        t = 0.0
        for i in np.where(assign[:, j] == 1)[0]:
            starts[i] = t
            t += duration[i]

    df = pd.DataFrame({
        "task": list(range(n_tasks)),
        "resource": assign.argmax(axis=1),
        "start": starts,
        "duration": duration,
        "cost": [cost[i, assign[i].argmax()] for i in range(n_tasks)],
    })
    return df
