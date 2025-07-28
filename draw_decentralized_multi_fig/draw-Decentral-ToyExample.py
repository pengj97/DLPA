import matplotlib.pyplot as plt
import os
import sys
import re
sys.path.append('..')
from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path

colors = ['green', 'olive',  'orange', 'blue', 'purple', 'grey', 'gold', 'red']
markers = ['h', '^', 'v',  's', 'x', 'o', '>',  'D']

# colors = ['green', 'olive',  'orange', 'blue', 'purple', 'grey', 'gold', 'skyblue', 'red']
# markers = ['h', '^', 'v',  's', 'x', 'o', '>', '<', 'D']

# colors = ['green', 'olive',  'orange', 'blue',  'red']
# markers = ['h', '^', 'v',  's',  '>']


# lr_ctrl = ''
lr_ctrl = '_invSqrtLR'
# lr_ctrl = '_ladderLR'

# task_name = 'NeuralNetwork'
task_name = 'SR'
# task_name = 'ResNet18'

# graph_name = 'Fan_n=10_b=1'
# graph_name = 'Complete_n=10_b=1'
# graph_name = 'ER_n=10_b=1_p=0.7_seed=300'
# graph_name = 'Line_n=10_b=1'
# graph_name = 'TwoCastle_k=5_b=1_seed=40'
# graph_name = 'Centralized_n=10_b=1'
# graph_name = 'Octopus_head=5_headb=0_handb=5'
# graph_name = 'UnconnectedRegularLine_n=10_b=1'
# graph_name = 'Fan_n=10_b=2'



attack_name = 'label_flipping'
# attack_name = 'furthest_label_flipping'
# attack_name = 'label_random'

FONTSIZE = 25

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(__FILE_DIR__, os.path.pardir, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)


def draw_mnist():
    # datasets = ['mnist', 'cifar10']
    dataset = 'mnist'


    aggregations = [
        ('meanW', 'Baseline'), 
        ('no_communication', 'No Comm.'),
        # ('median', 'CooMed'),
        # ('geometric_median', 'GeoMed'), 
        # ('Krum', 'Krum'), 
        ('trimmed_mean', 'TriMean'),
        ('FABA', 'FABA'), 
        ('CC', 'CC'),
        ('SCClip', 'CG'),
        # ('SCClip_T', 'SCC-T'),
        ('IOS', 'IOS'), 
        # ('bulyan', 'Bulyan'),
        # ('remove_outliers', 'Cutter'),
        # ('LFighter', 'LFighter'),
        ('meanW', 'WeiMean'), 
    ]
    partition_names = [
        # ('iidPartition', 'IID'),
        # ('DirichletPartition_alpha=1', 'Mild Noniid'),
        ('LabelSeperation', 'Noniid')
    ]
    graph_names = [
        ('RegularComplete_n=10_b=1', 'Lollipop graph'),
        ('Fan_n=10_b=1', 'Fan graph'),
    ]

    pic_name = 'decentralized_' + task_name + '_' + dataset + '_' + attack_name + lr_ctrl + '_toyexample' 

    fig, axes = plt.subplots(1, len(graph_names), figsize=(14, 8), sharex=True, sharey=True)

    axes[0].set_ylabel('Accuracy', fontsize=FONTSIZE)
    axes[0].set_ylim(0.6, 0.93)


    taskname = task_name + '_' + dataset
    for i in range(len(graph_names)):
        # axes[i].set_title(partition_names[0][1] + ' (MNIST)', fontsize=FONTSIZE)
        axes[i].set_title(graph_names[i][1], fontsize=FONTSIZE)
        axes[i].set_xlabel('iterations', fontsize=FONTSIZE)
        axes[i].tick_params(labelsize=FONTSIZE)
        axes[i].grid('on')
        for agg_index, (agg_code_name, agg_show_name) in enumerate(aggregations):
            color = colors[agg_index]
            marker = markers[agg_index]
            if agg_code_name == 'CC' or agg_code_name == 'SCClip':
                # agg_code_name += '_tau=0.1'
                agg_code_name += '_tau=0.03'
            if agg_show_name == 'Baseline':
                file_name = 'DSGD_baseline_meanW' + lr_ctrl
                file_path = [taskname, 'ER_n=10_b=0_p=0.7_seed=300', partition_names[0][0]]
            else:
                file_name = 'DSGD_' + attack_name + '_' + agg_code_name + lr_ctrl
                file_path = [taskname, graph_names[i][0], partition_names[0][0]]
            record = load_file_in_cache(file_name, path_list=file_path)
            acc_path = record['acc_path']
            x_axis = [r*record['display_interval']
                        for r in range(record['rounds']+1)]
            axes[i].plot(x_axis, acc_path, '-', color=color, marker=marker, label=agg_show_name, markevery=20, linewidth=4, markersize=10)
    
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=FONTSIZE, markerscale=2)
    leg_lines = leg.get_lines()
    for i in range(len(leg_lines)):
        plt.setp(leg_lines[i], linewidth=5.0)
    
    plt.subplots_adjust(top=0.91, bottom=0.30, left=0.125, right=1, hspace=0.27, wspace=0.2)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    dir_png_path = os.path.join(file_dir, 'pic', 'png')
    dir_pdf_path = os.path.join(file_dir, 'pic', 'pdf')

    if not os.path.isdir(dir_pdf_path):
        os.makedirs(dir_pdf_path)
    if not os.path.isdir(dir_png_path):
        os.makedirs(dir_png_path)

    suffix = ''
    pic_png_path = os.path.join(dir_png_path, pic_name + suffix + '.png')
    pic_pdf_path = os.path.join(dir_pdf_path, pic_name + suffix + '.pdf')
    plt.savefig(pic_png_path, format='png', bbox_inches='tight')
    plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_mnist()
