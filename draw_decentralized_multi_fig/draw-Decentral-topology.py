import os
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
from ByrdLab.graph import TwoCastle, FanGraph, UnconnectedRegularLineGraph

# Create figure
SCALE = 1.4
fig = plt.figure(figsize=(SCALE * 8, SCALE * 3))

# Plot (a)
ax1 = plt.subplot(1, 3, 1)
graph = TwoCastle(k=5, byzantine_size=1, seed=40)
graph.show(reverse=True, as_subplot=True, layout='circular', rotate=True, angle_degrees=-53)
ax1.axis("off")

# Plot (b)
ax3 = plt.subplot(1, 3, 2)
graph = UnconnectedRegularLineGraph(node_size=10, byzantine_size=1)
graph.show(reverse=True, as_subplot=True, layout='circular', rotate=True, angle_degrees=-53)
ax3.axis("off")

# Plot (c)
ax2 = plt.subplot(1, 3, 3)
graph = FanGraph(node_size=10, byzantine_size=1)
graph.show(as_subplot=True, layout='circular', rotate=True, angle_degrees=-53)
ax2.axis("off")



# Adjust spacing
fig.subplots_adjust(wspace=0.1, bottom=0.2)

# Add bold vertical lines between subplots
# fig.add_artist(plt.Line2D([0.38, 0.38], [0.15, 0.9], color='black', linewidth=1, transform=fig.transFigure, clip_on=False))
# fig.add_artist(plt.Line2D([0.65, 0.65], [0.15, 0.9], color='black', linewidth=1, transform=fig.transFigure, clip_on=False))

# Add subplot labels (a), (b), (c) under each plot
label_props = dict(fontsize=20, ha='center', va='center')
fig.text(0.2465, 0.1, '(a)', **label_props)
fig.text(0.5135, 0.1, '(b)', **label_props)
fig.text(0.78, 0.1, '(c)', **label_props)

# Save figure
output_dir = 'pic'
os.makedirs(output_dir, exist_ok=True)
pic_path = os.path.join(output_dir, 'draw_decentralized_graph.pdf')
plt.savefig(pic_path, format='pdf', bbox_inches='tight')
plt.show()
