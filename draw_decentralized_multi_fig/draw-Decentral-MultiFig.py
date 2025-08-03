import matplotlib.pyplot as plt
import os
import sys
import re
sys.path.append('..')
from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path

colors = ['green',  'orange', 'blue', 'purple', 'grey', 'gold', 'olive', 'red']
markers = ['h', '^', 'v',  's', 'x', 'o', '>',  'D']

lr_ctrl = '_invSqrtLR'
task_name = 'ResNet18'


# graph_name = 'TwoCastle_k=5_b=1_seed=40'
# graph_name = 'UnconnectedRegularLine_n=10_b=1'
# graph_name = 'Fan_n=10_b=2'
graph_name = 'Fan_n=10_b=1'




attack_name = 'label_flipping'
# attack_name = 'dynamic_label_flipping'

FONTSIZE = 50


__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(__FILE_DIR__, os.path.pardir, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)


def draw_mnist_acc_ce():

    dataset = 'cifar100'

    aggregations = [
        ('meanW', 'Baseline'), 
        ('trimmed_mean', 'TriMean'),
        ('FABA', 'FABA'), 
        ('CC', 'CC'),
        ('SCClip', 'CG'),
        ('IOS', 'IOS'), 
        ('LFighter', 'LFighter'),
        ('meanW', 'WeiMean'), 
    ]
    partition_names = [
        ('iidPartition', 'IID'),
        ('DirichletPartition_alpha=1', 'Mild Noniid'),
        ('LabelSeperation', 'Noniid')
    ]

    pic_name = 'decentralized_' + task_name + '_' + dataset + '_' + graph_name + '_' + attack_name + lr_ctrl
    fig, axes = plt.subplots(2, len(partition_names), figsize=(23, 16), sharex=True, sharey='row')

    axes[0][0].set_ylabel('Accuracy', fontsize=FONTSIZE)
    axes[0][0].set_ylim(0.15, 0.75)


    axes[1][0].set_ylabel('Consensus', fontsize=FONTSIZE)
    axes[1][0].set_ylim(1e-7, 1e-3)
    axes[1][0].set_yscale('log')



    taskname = task_name + '_' + dataset
    for i in range(len(partition_names)):
        for n in range(2):
            axes[n][i].set_title(partition_names[i][1] + f' ({dataset.upper()})', fontsize=FONTSIZE)
            axes[1][i].set_xlabel('iterations', fontsize=FONTSIZE)
            axes[n][i].tick_params(labelsize=FONTSIZE)
            axes[n][i].grid('on')
        for agg_index, (agg_code_name, agg_show_name) in enumerate(aggregations):
            color = colors[agg_index]
            marker = markers[agg_index]
            if agg_code_name == 'CC':
                agg_code_name += '_tau=100'
            elif agg_code_name == 'SCClip':
                agg_code_name += '_tau=0.1'
            if agg_show_name == 'Baseline':
                file_name = 'DSGD_baseline_meanW' + lr_ctrl
                if graph_name == 'UnconnectedRegularLine_n=10_b=1':
                    baseline_graph_name = 'Line_n=10_b=0'
                elif graph_name == 'Fan_n=10_b=2':
                    baseline_graph_name = 'Fan_n=10_b=0'
                else:
                    baseline_graph_name = re.sub(r'b=\d+', 'b=0', graph_name)
                file_path = [taskname, baseline_graph_name, partition_names[i][0]]
            else:
                file_name = 'DSGD_' + attack_name + '_' + agg_code_name + lr_ctrl
                file_path = [taskname, graph_name, partition_names[i][0]]
            record = load_file_in_cache(file_name, path_list=file_path)
            print(record['lr'])
            acc_path = record['acc_path']
            print(f'graph: {graph_name} agg_code_name: {agg_code_name}, acc_path: {acc_path[-1]}')
            ce_path = record['consensus_error_path']
            x_axis = [r*record['display_interval']
                        for r in range(record['rounds']+1)]
            axes[0][i].plot(x_axis, acc_path, '-', color=color, marker=marker, label=agg_show_name, markevery=20, linewidth=4, markersize=10)
            axes[1][i].plot(x_axis, ce_path, '-', color=color, marker=marker, label=agg_show_name, markevery=20, linewidth=4, markersize=10)

    for n in range(2):
        handles, labels = axes[n][0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=FONTSIZE, markerscale=2)
        leg_lines = leg.get_lines()
        for i in range(len(leg_lines)):
            plt.setp(leg_lines[i], linewidth=5.0)
    
    plt.subplots_adjust(top=0.91, bottom=0.30, left=0.01, right=1, hspace=0.19, wspace=0.2)

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
    draw_mnist_acc_ce()
