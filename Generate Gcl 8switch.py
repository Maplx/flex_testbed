# main_8switch.py
# Updated to use real partition output and 8-switch topology

import pandas as pd
import z3

# Define edges from 8-switch topology
edges = {
    ('S1', 'S2'): 0, ('S2', 'S1'): 1,
    ('S2', 'S3'): 2, ('S3', 'S2'): 3,
    ('S3', 'S4'): 4, ('S4', 'S3'): 5,
    ('S1', 'S5'): 6, ('S5', 'S1'): 7,
    ('S2', 'S6'): 8, ('S6', 'S2'): 9,
    ('S3', 'S7'): 10, ('S7', 'S3'): 11,
    ('S4', 'S8'): 12, ('S8', 'S4'): 13,
    ('S5', 'S6'): 14, ('S6', 'S5'): 15,
    ('S6', 'S7'): 16, ('S7', 'S6'): 17,
    ('S7', 'S8'): 18, ('S8', 'S7'): 19
}

reverse_edges = {v: k for k, v in edges.items()}

# Load partition and flow definitions
sched_df = pd.read_csv("partition_schedule.csv")
flows_df = pd.read_csv("flows_information_with_nodes.csv")

flows = []
schedule = {}
applications = set()
link_ids = set()

for idx, row in flows_df.iterrows():
    app_id = row['App ID']
    flow_id = row['Flow ID']
    period = int(row['Period'])
    path = row['Node Path'].split(" -> ")
    tx_links = [edges[(path[i], path[i+1])] for i in range(len(path)-1)]
    flows.append({"app_id": app_id, "flow_id": flow_id, "links": tx_links, "period": period})
    applications.add(app_id)
    link_ids.update(tx_links)

# Parse schedule
timeslots = sched_df.index
for t in timeslots:
    for col in sched_df.columns:
        if "Link" in col:
            l = int(col.split()[-1])
            app = int(sched_df.loc[t, col])
            schedule[(t, l)] = app

applications = sorted(applications)
link_ids = sorted(link_ids)

# Z3 Model
solver = z3.Solver()
flow_pcp = {(f["app_id"], f["flow_id"]): z3.Int(f"flow_pcp_{f['app_id']}_{f['flow_id']}") for f in flows}
slot_queue = {(l, t): z3.Int(f"slot_queue_{l}_{t}") for l in link_ids for t in timeslots}
pcp_queue = {l: z3.Array(f"pcp_queue_{l}", z3.IntSort(), z3.IntSort()) for l in link_ids}

# PCP ∈ [1,7], Queue ∈ [1,7], Slot Queue ∈ [1,7]
for (a, f), pcp in flow_pcp.items():
    solver.add(z3.And(pcp >= 1, pcp <= 7))

for l in link_ids:
    for pcp in range(8):
        solver.add(z3.And(pcp_queue[l][pcp] >= 1, pcp_queue[l][pcp] <= 7))
    solver.add(z3.Distinct([pcp_queue[l][pcp] for pcp in range(1, 8)]))

for (t, l), app in schedule.items():
    if app == -1:
        continue
    for f in flows:
        if f['app_id'] == app and l in f['links']:
            solver.add(slot_queue[l, t] == z3.Select(pcp_queue[l], flow_pcp[(app, f['flow_id'])]))

# Solve and extract results
if solver.check() == z3.sat:
    model = solver.model()
    with open("sche.txt", "w") as f:
        f.write("Timeslot to Queue mapping (slot_queue):\n")
        for l in link_ids:
            f.write(f"Link {l}:\n")
            for t in timeslots:
                if schedule.get((t, l), -1) == -1:
                    f.write(f"  Timeslot {t}: Queue 0\n")
                else:
                    q_expr = model.eval(slot_queue[l, t], model_completion=True)
                    q = q_expr.as_long() if q_expr is not None else 0
                    f.write(f"  Timeslot {t}: Queue {q}\n")
else:
    print("No feasible schedule found")