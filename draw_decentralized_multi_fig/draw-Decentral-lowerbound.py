import os
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
from ByrdLab.graph import TwoCastle, FanGraph, UnconnectedRegularLineGraph, LowerBoundGraph

# Create figure
SCALE = 1.4
fig = plt.figure(figsize=(SCALE * 6, SCALE * 3))

# # Plot (a)
# ax1 = plt.subplot(1, 2, 1)
# graph = LowerBoundGraph(node_size=10, byzantine_size=5)
# graph.show(rotate=True, as_subplot=True, show_label=True, angle_degrees=15,
#            label_dict={0:1, 1:1, 2:1, 3:1, 4:1, 5:2, 6:2, 7:2, 8:2, 9:2})
# ax1.axis("off")

# # Plot (b)
# ax2 = plt.subplot(1, 2, 2)
# graph = LowerBoundGraph(node_size=10, byzantine_size=5)
# graph.show(rotate=True, as_subplot=True, show_label=True, angle_degrees=15,
#            label_dict={0:2, 1:1, 2:2, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1})
# ax2.axis("off")


# Plot (a)
ax1 = plt.subplot(1, 2, 1)
graph = LowerBoundGraph(node_size=8, byzantine_size=4)
graph.show(rotate=True, as_subplot=True, show_label=True, angle_degrees=15,
           label_dict={0:1, 1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:2})
ax1.axis("off")

# Plot (b)
ax2 = plt.subplot(1, 2, 2)
graph = LowerBoundGraph(node_size=8, byzantine_size=4)
graph.show(rotate=True, as_subplot=True, show_label=True, angle_degrees=15,
           label_dict={0:2, 1:1, 2:2, 3:1, 4:1, 5:1, 6:1, 7:1})
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
pic_path = os.path.join(output_dir, 'draw_decentralized_lowerbound.pdf')
plt.savefig(pic_path, format='pdf', bbox_inches='tight')
plt.show()
