# format_gcl_8switch.py
import csv

TIME_SLOT_SIZE = 50  # microseconds per slot
CYCLE_TIME = 5000  # microseconds (e.g., 100 slots × 50μs)

slot_queue = {}

with open("sche.txt", "r") as f:
    lines = f.readlines()

current_link = None
for line in lines:
    line = line.strip()
    if line.startswith("Link"):
        current_link = int(line.split()[1].rstrip(":"))
        slot_queue[current_link] = {}
    elif line.startswith("Timeslot"):
        try:
            t = int(line.split()[1].rstrip(":"))
            q = int(line.split()[-1])
            slot_queue[current_link][t] = q
        except ValueError:
            continue  # skip malformed lines
        q = int(line.split()[-1])
        slot_queue[current_link][t] = q

import os
os.makedirs("output", exist_ok=True)

with open("output/gcl.csv", "w", newline="") as file:
    fieldnames = ["link", "queue", "start", "end", "cycle"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for link, ts in slot_queue.items():
        for t, q in ts.items():
            writer.writerow({
                "link": link,
                "queue": q,
                "start": t * TIME_SLOT_SIZE,
                "end": (t+1) * TIME_SLOT_SIZE,
                "cycle": CYCLE_TIME
            })
