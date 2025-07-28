import os
import sys
sys.path.append('..')
import matplotlib.pyplot as plt

from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path


from ByrdLab.graph import RegularCompleteGraph, FanGraph

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(__FILE_DIR__, os.path.pardir, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)

taskname = 'SR_mnist'
W = 10
P = 1

topologies = ['RegularComplete', 'Fan']
aggregations = [
        ('trimmed_mean', 'TriMean'),
        ('FABA', 'FABA'), 
        ('CC_tau=0.03', 'CC'),
        ('SCClip_tau=0.03', 'CG'),
        ('IOS', 'IOS'), 
        ('meanW', 'WeiMean'), 
]
partition_names = ('LabelSeperation', 'Noniid')
attacks = 'label_flipping'

FONTSIZE = 25
colors = [ 'orange', 'blue', 'purple', 'grey', 'gold',  'red']
markers = [ 'v',  's', 'x', 'o', '>',  'D']

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 14))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1])

# Plot (a)
ax1 = plt.subplot(gs[0, 0])
graph = RegularCompleteGraph(node_size=10, byzantine_size=1)
graph.show(rotate=True, as_subplot=True, layout='lollipop', angle_degrees=309.5)
ax1.axis("off")

# Plot (b)
ax2 = plt.subplot(gs[0, 1])
graph = FanGraph(node_size=10, byzantine_size=1)
graph.show(rotate=True, as_subplot=True, layout='circular', angle_degrees=308)
ax2.axis("off")

# Accuracy plots
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1])
axes = [ax3, ax4]
axes[0].set_ylabel('Accuracy', fontsize=FONTSIZE)
axes[0].set_ylim(0.6, 0.83)
axes[1].set_ylim(0.6, 0.83)
axes[1].tick_params(labelleft=False)  # 不显示第二个子图的y轴刻度标签



for i in range(len(topologies)):
    # axes[i].set_title(f'{topologies[i]} graph', fontsize=FONTSIZE)
    axes[i].set_xlabel('iterations', fontsize=FONTSIZE)
    axes[i].tick_params(labelsize=FONTSIZE)
    axes[i].grid(True)  # 使用True而不是'on'
    for agg_index, (agg_code_name, agg_show_name) in enumerate(aggregations):
        color = colors[agg_index]
        marker = markers[agg_index]
        file_path = [taskname, f'{topologies[i]}_n={W}_b={P}', partition_names[0]]
        file_name = f'DSGD_{attacks}_{agg_code_name}_invSqrtLR'
        record = load_file_in_cache(file_name, path_list=file_path)

        acc_path = record['acc_path']
        x_axis = [r*record['display_interval'] for r in range(record['rounds']+1)]
        axes[i].plot(x_axis, acc_path, '-', color=color, marker=marker, 
                    label=agg_show_name, markevery=10, linewidth=4, markersize=10)
        
# Add subplot labels (a), (b) under each plot
label_props = dict(fontsize=FONTSIZE, ha='center', va='center')
fig.text(0.30, 0.13, '(a)', **label_props)
fig.text(0.78, 0.13, '(b)', **label_props)

fig.text(0.29, 0.575, 'Lollipop graph', **label_props)
fig.text(0.78, 0.575, 'Fan graph', **label_props)

handles, labels = axes[0].get_legend_handles_labels() 
leg = fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=FONTSIZE, markerscale=2)
leg_lines = leg.get_lines()
for i in range(len(leg_lines)):
    plt.setp(leg_lines[i], linewidth=5.0)   

plt.subplots_adjust(top=1,
                   bottom=0.22,
                   left=0.1,
                   right=0.97,
                   hspace=0.13,
                   wspace=0.2)

# plt.savefig(f'toyexample_{partition_param}_{attacks.__name__}_multiclass.pdf', format='pdf', bbox_inches='tight')
plt.savefig(f'toyexample.pdf', format='pdf', bbox_inches='tight')

plt.show()