import matplotlib.pyplot as plt
import os
import sys
import re
sys.path.append('..')
from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path

colors = ['green',  'orange', 'blue', 'purple', 'grey', 'gold', 'olive', 'red']
markers = ['h', '^', 'v',  's', 'x', 'o', '>',  'D']

# colors = ['green', 'olive',  'orange', 'blue', 'purple', 'grey', 'gold', 'skyblue', 'red']
# markers = ['h', '^', 'v',  's', 'x', 'o', '>', '<', 'D']

# colors = ['green', 'olive',  'orange', 'blue',  'red']
# markers = ['h', '^', 'v',  's',  '>']


# lr_ctrl = ''
lr_ctrl = '_invSqrtLR'
# lr_ctrl = '_ladderLR'

# task_name = 'NeuralNetwork'
# task_name = 'SR'
task_name = 'ResNet18'

# graph_name = 'Fan_n=10_b=1'
# graph_name = 'Complete_n=10_b=1'
# graph_name = 'ER_n=10_b=1_p=0.7_seed=300'
# graph_name = 'Line_n=10_b=1'
# graph_name = 'Centralized_n=10_b=1'
# graph_name = 'Octopus_head=5_headb=0_handb=5'

# graph_name = 'TwoCastle_k=5_b=1_seed=40'
# graph_name = 'UnconnectedRegularLine_n=10_b=1'
# graph_name = 'Fan_n=10_b=2'
graph_name = 'Fan_n=10_b=1'




# attack_name = 'label_flipping'
attack_name = 'furthest_label_flipping'
# attack_name = 'label_random'

# FONTSIZE = 25
FONTSIZE = 50


__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(__FILE_DIR__, os.path.pardir, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)

# def draw():
#     # datasets = ['mnist', 'cifar10']
#     datasets = ['mnist']


#     aggregations = [
#         ('mean', 'Mean (without attacks)'), 
#         ('mean', 'Mean'), 
#         # ('median', 'CooMed'),
#         # ('geometric_median', 'GeoMed'), 
#         # ('Krum', 'Krum'), 
#         ('trimmed_mean', 'TriMean'),
#         # ('SCClip', 'SCC'),
#         # ('SCClip_T', 'SCC-T'),
#         ('faba', 'IOS/FABA'), 
#         ('CC', 'CC'),
#         # ('IOS', r'\textbf{IOS (ours)}'), 
#         # ('bulyan', 'Bulyan'),
#         # ('remove_outliers', 'Cutter'),
#     ]
#     partition_names = [
#         ('iidPartition', 'IID'),
#         ('DirichletPartition_alpha=1', 'Mild Noniid'),
#         ('LabelSeperation', 'Noniid')
#     ]

#     pic_name = 'centralized_' + task_name + '_' + graph_name + '_' + attack_name

#     fig, axes = plt.subplots(2, len(partition_names), figsize=(21, 12), sharex=True, sharey=True)
#     axes[0][0].set_ylabel('Accuracy', fontsize=FONTSIZE-5)
#     axes[1][0].set_ylabel('Accuracy', fontsize=FONTSIZE-5)
#     for l in range(len(datasets)):
#         taskname = task_name + '_' + datasets[l]
#         for i in range(len(partition_names)):
#             axes[l][i].set_title(partition_names[i][1] + f' ({datasets[l]})'.upper(), fontsize=FONTSIZE)
#             axes[l][i].set_xlabel('iterations', fontsize=FONTSIZE)
#             axes[l][i].tick_params(labelsize=FONTSIZE - 5)
#             axes[l][i].grid('on')
#             for agg_index, (agg_code_name, agg_show_name) in enumerate(aggregations):
#                 color = colors[agg_index]
#                 marker = markers[agg_index]
#                 if partition_names[i][0] == 'iidPartition' and agg_code_name == 'CC':
#                     agg_code_name += '_tau=0.1'
#                 elif agg_code_name == 'CC':
#                     agg_code_name += '_tau=0.3'

#                 if agg_show_name == 'Mean (without attacks)':
#                     file_name = 'CSGD_baseline_mean'
#                     file_path = [taskname, 'Centralized_n=10_b=0', partition_names[i][0]]
#                 else:
#                     file_name = 'CSGD_' + attack_name + '_' + agg_code_name + ''
#                     file_path = [taskname, graph_name, partition_names[i][0]]

#                 record = load_file_in_cache(file_name, path_list=file_path)
#                 acc_path = record['acc_path']

#                 x_axis = [r*record['display_interval']
#                             for r in range(record['rounds']+1)]

#                 axes[l][i].plot(x_axis, acc_path, '-', color=color, marker=marker, label=agg_show_name, markevery=20)

#     handles, labels = axes[0][0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=FONTSIZE - 5)

#     plt.subplots_adjust(top=0.91, bottom=0.1, left=0.125, right=0.9, hspace=0.255, wspace=0.2)


#     file_dir = os.path.dirname(os.path.abspath(__file__))
#     dir_png_path = os.path.join(file_dir, 'pic', 'png')
#     dir_pdf_path = os.path.join(file_dir, 'pic', 'pdf')

#     if not os.path.isdir(dir_pdf_path):
#         os.makedirs(dir_pdf_path)
#     if not os.path.isdir(dir_png_path):
#         os.makedirs(dir_png_path)

#     suffix = ''
#     pic_png_path = os.path.join(dir_png_path, pic_name + suffix + '.png')
#     pic_pdf_path = os.path.join(dir_pdf_path, pic_name + suffix + '.pdf')
#     plt.savefig(pic_png_path, format='png', bbox_inches='tight')
#     plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')
#     plt.show()

# def draw_mnist():
#     # datasets = ['mnist', 'cifar10']
#     dataset = 'mnist'


#     aggregations = [
#         # ('meanW', 'Mean (without attacks)'), 
#         # ('median', 'CooMed'),
#         # ('geometric_median', 'GeoMed'), 
#         # ('Krum', 'Krum'), 
#         ('trimmed_mean', 'TriMean'),
#         ('CC', 'CC'),
#         ('SCClip', 'SCC'),
#         # ('SCClip_T', 'SCC-T'),
#         ('FABA', 'FABA'), 
#         ('IOS', 'IOS'), 
#         # ('bulyan', 'Bulyan'),
#         # ('remove_outliers', 'Cutter'),
#         ('meanW', 'Weighted Mean'), 
#     ]
#     partition_names = [
#         ('iidPartition', 'IID'),
#         ('DirichletPartition_alpha=1', 'Mild Noniid'),
#         ('LabelSeperation', 'Noniid')
#     ]

#     pic_name = 'decentralized_' + task_name + '_' + dataset + '_' + graph_name + '_' + attack_name + lr_ctrl

#     fig, axes = plt.subplots(2, len(partition_names), figsize=(22, 7), sharex=True, sharey=True)
#     # fig, axes = plt.subplots(1, len(partition_names), figsize=(14, 7), sharex=True, sharey=True)

#     axes[0].set_ylabel('Accuracy', fontsize=FONTSIZE)
#     axes[0].set_ylim(0.6, 0.93)


#     taskname = task_name + '_' + dataset
#     for i in range(len(partition_names)):
#         axes[i].set_title(partition_names[i][1] + ' (MNIST)', fontsize=FONTSIZE)
#         axes[i].set_xlabel('iterations', fontsize=FONTSIZE)
#         axes[i].tick_params(labelsize=FONTSIZE)
#         axes[i].grid('on')
#         for agg_index, (agg_code_name, agg_show_name) in enumerate(aggregations):
#             color = colors[agg_index]
#             marker = markers[agg_index]
#             if partition_names[i][0] == 'iidPartition' and (agg_code_name == 'CC' or agg_code_name == 'SCClip'):
#                 # agg_code_name += '_tau=0.1'
#                 agg_code_name += '_tau=0.01'
#             elif agg_code_name == 'CC' or agg_code_name == 'SCClip':
#                 # agg_code_name += '_tau=0.3'
#                 agg_code_name += '_tau=0.03'
#             if agg_show_name == 'Mean (without attacks)':
#                 file_name = 'DSGD_baseline_meanW' + lr_ctrl
#                 file_path = [taskname, 'ER_n=10_b=0_p=0.7_seed=300', partition_names[i][0]]
#             else:
#                 file_name = 'DSGD_' + attack_name + '_' + agg_code_name + lr_ctrl
#                 file_path = [taskname, graph_name, partition_names[i][0]]
#             record = load_file_in_cache(file_name, path_list=file_path)
#             acc_path = record['acc_path']
#             x_axis = [r*record['display_interval']
#                         for r in range(record['rounds']+1)]
#             axes[i].plot(x_axis, acc_path, '-', color=color, marker=marker, label=agg_show_name, markevery=20, linewidth=4, markersize=10)
    
#     handles, labels = axes[0].get_legend_handles_labels()
#     leg = fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=FONTSIZE, markerscale=2)
#     leg_lines = leg.get_lines()
#     for i in range(len(leg_lines)):
#         plt.setp(leg_lines[i], linewidth=5.0)
    
#     plt.subplots_adjust(top=0.91, bottom=0.27, left=0.125, right=1, hspace=0.27, wspace=0.2)

#     file_dir = os.path.dirname(os.path.abspath(__file__))
#     dir_png_path = os.path.join(file_dir, 'pic', 'png')
#     dir_pdf_path = os.path.join(file_dir, 'pic', 'pdf')

#     if not os.path.isdir(dir_pdf_path):
#         os.makedirs(dir_pdf_path)
#     if not os.path.isdir(dir_png_path):
#         os.makedirs(dir_png_path)

#     suffix = ''
#     pic_png_path = os.path.join(dir_png_path, pic_name + suffix + '.png')
#     pic_pdf_path = os.path.join(dir_pdf_path, pic_name + suffix + '.pdf')
#     plt.savefig(pic_png_path, format='png', bbox_inches='tight')
#     plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')
#     plt.show()

def draw_mnist_acc_ce():
    # dataset = 'mnist'
    # dataset = 'cifar10'
    dataset = 'cifar100'

    aggregations = [
        ('meanW', 'Baseline'), 
        # ('no_communication', 'No Comm.'),
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
        ('LFighter', 'LFighter'),
        ('meanW', 'WeiMean'), 
    ]
    partition_names = [
        ('iidPartition', 'IID'),
        ('DirichletPartition_alpha=1', 'Mild Noniid'),
        ('LabelSeperation', 'Noniid')
    ]

    pic_name = 'decentralized_' + task_name + '_' + dataset + '_' + graph_name + '_' + attack_name + lr_ctrl

    # fig, axes = plt.subplots(2, len(partition_names), figsize=(22, 14), sharex=True, sharey='row')
    # fig, axes = plt.subplots(1, len(partition_names), figsize=(14, 7), sharex=True, sharey=True)
    # fig, axes = plt.subplots(2, len(partition_names), figsize=(22, 14), sharex=True, sharey='row')
    # fig, axes = plt.subplots(2, len(partition_names), figsize=(22, 14), sharex=True, sharey='row')
    fig, axes = plt.subplots(2, len(partition_names), figsize=(23, 16), sharex=True, sharey='row')


    # fig, axes = plt.subplots(2, 3, figsize=(22, 14), sharex=True)


    axes[0][0].set_ylabel('Accuracy', fontsize=FONTSIZE)
    axes[0][0].set_ylim(0.15, 0.75)
    # axes[0][1].set_ylim(0.3, 0.93)
    # axes[0][2].set_ylim(0.3, 0.93)

    axes[1][0].set_ylabel('Consensus', fontsize=FONTSIZE)
    # axes[1][0].set_ylim(1e-9, 1e-4)
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
            # if partition_names[i][0] == 'iidPartition' and (agg_code_name == 'CC' or agg_code_name == 'SCClip'):
            #     # agg_code_name += '_tau=0.1'
            #     # agg_code_name += '_tau=0.01'
            #     agg_code_name += '_tau=100'

            # elif agg_code_name == 'CC' or agg_code_name == 'SCClip':
            #     # agg_code_name += '_tau=0.3'
            #     # agg_code_name += '_tau=0.03'
            #     agg_code_name += '_tau=300'

            if partition_names[i][0] == 'iidPartition': 
                if agg_code_name == 'CC':
                # agg_code_name += '_tau=0.1'
                # agg_code_name += '_tau=0.01'
                    agg_code_name += '_tau=100'
                elif agg_code_name == 'SCClip':
                    # agg_code_name += '_tau=30'
                    # agg_code_name += '_tau=0.1'
                    # agg_code_name += '_tau=0.01'
                    agg_code_name += '_tau=0.1'


            elif agg_code_name == 'CC':
                # agg_code_name += '_tau=0.3'
                # agg_code_name += '_tau=0.03'
                # agg_code_name += '_tau=200'
                # agg_code_name += '_tau=300'
                agg_code_name += '_tau=100'


            elif agg_code_name == 'SCClip':
                # agg_code_name += '_tau=100'
                # agg_code_name += '_tau=0.3'
                # agg_code_name += '_tau=0.03'
                agg_code_name += '_tau=0.1'




            if agg_show_name == 'Baseline':
                file_name = 'DSGD_baseline_meanW' + lr_ctrl
                if graph_name == 'UnconnectedRegularLine_n=10_b=1':
                    # baseline_graph_name = 'UnconnectedRegularLine_n=9_b=0'
                    baseline_graph_name = 'Line_n=10_b=0'
                elif graph_name == 'Fan_n=10_b=2':
                    # baseline_graph_name = 'Fan_n=8_b=0'
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
    # draw_mnist()
    draw_mnist_acc_ce()
