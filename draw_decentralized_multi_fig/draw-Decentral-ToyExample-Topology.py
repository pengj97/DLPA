import os
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
from ByrdLab.graph import RegularCompleteGraph, FanGraph

# Create figure
SCALE = 1.8
fig = plt.figure(figsize=(SCALE * 6, SCALE * 3.5))

# Plot (a)
ax1 = plt.subplot(1, 2, 1)
graph = RegularCompleteGraph(node_size=10, byzantine_size=1)
graph.show(rotate=True, as_subplot=True, layout='lollipop', angle_degrees=309.5)
ax1.axis("off")

# Plot (b)
ax2 = plt.subplot(1, 2, 2)
graph = FanGraph(node_size=10, byzantine_size=1)
graph.show(rotate=True, as_subplot=True, layout='circular', angle_degrees=308)
ax2.axis("off")


# Adjust spacing
fig.subplots_adjust(wspace=0.1, bottom=0.2)

# Add bold vertical lines between subplots
# fig.add_artist(plt.Line2D([0.38, 0.38], [0.15, 0.9], color='black', linewidth=1, transform=fig.transFigure, clip_on=False))
# fig.add_artist(plt.Line2D([0.65, 0.65], [0.15, 0.9], color='black', linewidth=1, transform=fig.transFigure, clip_on=False))

# Add subplot labels (a), (b), (c) under each plot
label_props = dict(fontsize=20, ha='center', va='center')
fig.text(0.31, 0.15, '(a)', **label_props)
# fig.text(0.52, 0.05, '(b)', **label_props)
fig.text(0.72, 0.15, '(b)', **label_props)

# Save figure
output_dir = 'pic'
os.makedirs(output_dir, exist_ok=True)
pic_path = os.path.join(output_dir, 'draw_toyexample_topology.pdf')
plt.savefig(pic_path, format='pdf', bbox_inches='tight')
plt.show()
