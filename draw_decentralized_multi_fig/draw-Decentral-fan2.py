import os
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
from ByrdLab.graph import TwoCastle, FanGraph, UnconnectedRegularLineGraph

# Create figure
SCALE = 1.4
fig = plt.figure(figsize=(SCALE * 3, SCALE * 3))

# Plot
graph = FanGraph(node_size=10, byzantine_size=2)
graph.show(as_subplot=True, layout='circular', rotate=True, angle_degrees=-17.5)
plt.axis("off")



# Adjust spacing
# fig.subplots_adjust(wspace=0.1, bottom=0.2)

# Add bold vertical lines between subplots
# fig.add_artist(plt.Line2D([0.38, 0.38], [0.15, 0.9], color='black', linewidth=1, transform=fig.transFigure, clip_on=False))
# fig.add_artist(plt.Line2D([0.65, 0.65], [0.15, 0.9], color='black', linewidth=1, transform=fig.transFigure, clip_on=False))

# Add subplot labels (a), (b), (c) under each plot
# label_props = dict(fontsize=20, ha='center', va='center')
# fig.text(0.25, 0.05, '(a)', **label_props)
# fig.text(0.52, 0.05, '(b)', **label_props)
# fig.text(0.785, 0.05, '(c)', **label_props)

# Save figure
output_dir = 'pic'
os.makedirs(output_dir, exist_ok=True)
pic_path = os.path.join(output_dir, 'Fan_n=10_b=2.pdf')
plt.savefig(pic_path, format='pdf', bbox_inches='tight')
plt.show()
