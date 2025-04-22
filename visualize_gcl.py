# visualize_gcl.py
import pandas as pd
import matplotlib.pyplot as plt

# Load GCL CSV
df = pd.read_csv("output/gcl.csv")

# Create combined plot for all links
plt.figure(figsize=(12, 0.7 * len(df['link'].unique())))
y_positions = {link: i for i, link in enumerate(sorted(df['link'].unique(), reverse=True))}

colors = plt.cm.get_cmap("tab10", 8)  # up to 8 queues
queue_labels = {}
for _, row in df.iterrows():
    link = row['link']
    y = y_positions[link]
    width = (row['end'] - row['start']) / 50
    left = row['start'] / 50
    queue = row['queue']
    color = colors(queue % 8)
    plt.barh(y=y, width=width, left=left, height=0.6, color=color, alpha=0.7)
    plt.text(left + width/2, y, str(queue), va='center', ha='center', fontsize=7, color='black')
    queue_labels[queue] = color

plt.yticks(list(y_positions.values()), [f"Link {l}" for l in y_positions.keys()])
plt.xlabel("Time (μs, slot = 50μs)")
plt.title("Combined GCL Schedule for All Links")
handles = [plt.Rectangle((0,0),1,1, color=queue_labels[q]) for q in sorted(queue_labels)]
labels = [f"Queue {q}" for q in sorted(queue_labels)]
plt.legend(handles, labels, title="Queues", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("output/gcl_combined.png")
plt.close()