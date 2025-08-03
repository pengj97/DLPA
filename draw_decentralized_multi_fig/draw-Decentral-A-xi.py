import matplotlib.pyplot as plt
import os
import sys
import numpy as np
sys.path.append('..')
from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path

colors = [ 'green', 'red',  'orange', 'blue', 'purple']
markers = [ 'v', '^', '+', 's', 'x', 'o']

seed = 100
interval = 10
rounds = 100
FONTSIZE = 50

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(__FILE_DIR__, os.path.pardir, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)

method = 'DSGD'
lr = '_invSqrtLR'
graph_name = 'TwoCastle_k=5_b=1_seed=40'

def draw(task_name):

    dataset = 'cifar100'


    suffix_list = [
        ('_disturbance_A', 'Disturbance'),
        ('_hetero_list', 'Heterogeneity'),
    ]
    partition_names = [
        ('iidPartition', 'IID'),
        ('DirichletPartition_alpha=1', 'Mild Noniid'),
        ('LabelSeperation', 'Noniid')
    ]

    pic_name =   'decentralized_' + task_name + '_' + dataset + '_' + method + '_' + graph_name + '_A_hetero'

    fig, axes = plt.subplots(1, len(partition_names), figsize=(21, 10), sharex=True, sharey=True)
    axes[0].set_ylabel('Magnitude', fontsize=FONTSIZE)

    taskname = task_name + '_' + dataset
    for i in range(len(partition_names)):
        axes[i].set_title(partition_names[i][1] + f' ({dataset.upper()})', fontsize=FONTSIZE)
        axes[i].set_xlabel('iterations', fontsize=FONTSIZE)
        axes[i].tick_params(labelsize=FONTSIZE)
        axes[i].grid('on')
        for index, (suffix, label) in enumerate(suffix_list):
            color = colors[index]
            marker = markers[index]

            records = []
            if label == 'Heterogeneity':
                file_name = method + '_label_flipping_meanW' + lr + suffix + f'_seed={seed}' 
            elif label == 'Disturbance':
                file_name = method + '_label_flipping_meanW' + lr + suffix + f'_seed={seed}' 
            file_path = [taskname, graph_name, partition_names[i][0]]
        
            record = load_file_in_cache(file_name, path_list=file_path)
            records.append(record)


            records = np.array(records)
            mean_record = np.mean(records, axis=0)
            std_record = np.std(records, axis=0)
            
            x_axis = [r*interval for r in range(rounds+1)]
            axes[i].plot(x_axis, mean_record, '-', color=color, label=label, linewidth=4)
            axes[i].fill_between(x_axis, mean_record - std_record, mean_record + std_record, color=color, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=FONTSIZE)

    plt.subplots_adjust(top=1, bottom=0.37, left=0, right=1, hspace=0.1, wspace=0.13)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    dir_png_path = os.path.join(file_dir, 'pic', 'png')
    dir_pdf_path = os.path.join(file_dir, 'pic', 'pdf')

    if not os.path.isdir(dir_pdf_path):
        os.makedirs(dir_pdf_path)
    if not os.path.isdir(dir_png_path):
        os.makedirs(dir_png_path)

    pic_png_path = os.path.join(dir_png_path, pic_name + '.png')
    pic_pdf_path = os.path.join(dir_pdf_path, pic_name + '.pdf')
    plt.savefig(pic_png_path, format='png', bbox_inches='tight')
    plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw('ResNet18')
