from argsParser import args

from ByrdLab import FEATURE_TYPE
from ByrdLab.aggregation import (D_faba, D_trimmed_mean, D_meanW, D_no_communication, D_self_centered_clipping,
                                 D_centered_clipping, D_ios, D_LFighter)
from ByrdLab.attack import (label_flipping_mnist, dynamic_label_flipping)
from ByrdLab.decentralizedAlgorithm import DSGD, DSGD_under_DPA,  DSGD_with_LFighter_under_DPA
from ByrdLab.graph import CompleteGraph,  TwoCastle,  FanGraph, UnconnectedRegularLineGraph
from ByrdLab.library.cache_io import dump_file_in_cache
from ByrdLab.library.dataset import  mnist, cifar100
from ByrdLab.library.learnRateController import ladder_lr, one_over_sqrt_k_lr
from ByrdLab.library.partition import (LabelSeperation, TrivalPartition,
                                   iidPartition, DirichletMildPartition)
from ByrdLab.library.tool import log
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask
from ByrdLab.tasks.ResNet import ResNetTask

args.lr_ctrl = '1/sqrt k'


# run for decentralized algorithm
# -------------------------------------------
# define graph
# -------------------------------------------
if args.graph == 'complete':
    graph = CompleteGraph(node_size=10, byzantine_size=0)
elif args.graph == 'two-castle':
    graph = TwoCastle(k=5, byzantine_size=1, seed=40) # generate graph with 2k nodes
elif args.graph == 'fan':
    honest_size = 9
    byzantine_size = 1
    node_size = honest_size + byzantine_size
    graph = FanGraph(node_size=node_size, byzantine_size=byzantine_size)
elif args.graph == 'line':
    graph = UnconnectedRegularLineGraph(node_size=10, byzantine_size=1)
else:
    assert False, 'unknown graph'
if args.attack == 'none':
    graph = graph.honest_subgraph()

# ===========================================

# -------------------------------------------
# define learning task
# -------------------------------------------

data_package = mnist()
task = softmaxRegressionTask(data_package, batch_size=5000)

# data_package = cifar100()
# task = ResNetTask(data_package, batch_size=32)
# ===========================================


# -------------------------------------------
# define learning rate control rule
# -------------------------------------------
if args.lr_ctrl == 'constant':
    lr_ctrl = None
elif args.lr_ctrl == '1/sqrt k':
    lr_ctrl = one_over_sqrt_k_lr(a=1, b=1)
elif args.lr_ctrl == 'ladder':
    decreasing_iter_ls = [100, 200, 300, 400, 500, 600, 700, 800, 900]  
    proportion_ls = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    lr_ctrl = ladder_lr(decreasing_iter_ls, proportion_ls)
else:
    assert False, 'unknown lr-ctrl'

# ===========================================
    
    
# -------------------------------------------
# define data partition
# -------------------------------------------
if args.data_partition == 'trival':
    partition_cls = TrivalPartition
elif args.data_partition == 'iid':
    partition_cls = iidPartition
elif args.data_partition == 'noniid':
    partition_cls = LabelSeperation
elif args.data_partition == 'dirichlet_mild':
    partition_cls = DirichletMildPartition
else:
    assert False, 'unknown data-partition'
# ===========================================
    

# -------------------------------------------
# define aggregation
# -------------------------------------------
if args.aggregation == 'no-comm':
    aggregation = D_no_communication(graph)
elif args.aggregation == 'mean':
    aggregation = D_meanW(graph)
elif args.aggregation == 'ios':
    aggregation = D_ios(graph)
elif args.aggregation == 'trimmed-mean':
    aggregation = D_trimmed_mean(graph)
elif args.aggregation == 'faba':
    aggregation = D_faba(graph)
elif args.aggregation == 'cc':
    aggregation = D_centered_clipping(graph, threshold=0.03)
elif args.aggregation == 'scc':
    aggregation = D_self_centered_clipping(
        graph, threshold_selection='parameter', threshold=0.03)
elif args.aggregation == 'lfighter':
    aggregation = D_LFighter(graph)
else:
    assert False, 'unknown aggregation'

# ===========================================
    
# -------------------------------------------
# define attack
# -------------------------------------------
if args.attack == 'none':
    attack = None
elif args.attack == 'label_flipping':
    attack = label_flipping_mnist()
elif args.attack == 'dynamic_label_flipping':
    attack = dynamic_label_flipping()
else:
    assert False, 'unknown attack'

if args.attack == 'none':
    attack_name = 'baseline'
else:
    attack_name = attack.name

# ===========================================

workspace = []
mark_on_title = ''
fix_seed = not args.no_fixed_seed
seed = args.seed
record_in_file = not args.without_record
step_agg = args.step_agg

if 'label' in attack_name:
    if args.aggregation == 'lfighter':
        env = DSGD_with_LFighter_under_DPA(aggregation=aggregation, graph=graph, attack=attack, step_agg = step_agg,
           weight_decay=task.weight_decay, data_package=task.data_package,
           model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
           initialize_fn=task.initialize_fn,
           get_train_iter=task.get_train_iter,
           get_test_iter=task.get_test_iter,
           partition_cls=partition_cls, lr_ctrl=lr_ctrl,
           fix_seed=fix_seed, seed=seed,
           **task.super_params)
    else:
        env = DSGD_under_DPA(aggregation=aggregation, graph=graph, attack=attack, step_agg = step_agg,
               weight_decay=task.weight_decay, data_package=task.data_package,
               model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
               initialize_fn=task.initialize_fn,
               get_train_iter=task.get_train_iter,
               get_test_iter=task.get_test_iter,
               partition_cls=partition_cls, lr_ctrl=lr_ctrl,
               fix_seed=fix_seed, seed=seed,
               **task.super_params)
else:
    env = DSGD(aggregation=aggregation, graph=graph, attack=attack, step_agg = step_agg,
           weight_decay=task.weight_decay, data_package=task.data_package,
           model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
           initialize_fn=task.initialize_fn,
           get_train_iter=task.get_train_iter,
           get_test_iter=task.get_test_iter,
           partition_cls=partition_cls, lr_ctrl=lr_ctrl,
           fix_seed=fix_seed, seed=seed,
           **task.super_params)



title = '{}_{}_{}'.format(env.name, attack_name, aggregation.name)

if lr_ctrl != None:
    title = title + '_' + lr_ctrl.name
if mark_on_title != '':
    title = title + '_' + mark_on_title

data_package = task.data_package
super_params = task.super_params

# print the running information
print('=========================================================')
print('[Task] ' + task.name + ': ' + title)
print('=========================================================')
print('[Setting]')
print('{:12s} model={}'.format('[task]', task.model_name))
print('{:12s} dataset={} partition={}'.format(
    '[dataset]', data_package.name, env.partition_name))
print('{:12s} name={} aggregation={} attack={}'.format(
    '[Algorithm]', env.name, aggregation.name, attack_name))
print('{:12s} lr={} lr_ctrl={}, weight_decay={}'.format(
    '[Optimizer]', super_params['lr'], env.lr_ctrl.name, task.weight_decay))
print('{:12s} graph={}, slack_param={}, honest_size={}, byzantine_size={}'.format(
    '[Graph]', graph.name, args.slack_param, graph.honest_size, graph.byzantine_size))
print('{:12s} rounds={}, display_interval={}, total iterations={}'.format(
    '[Running]', env.rounds, env.display_interval, env.total_iterations))
print('{:12s} seed={}, fix_seed={}'.format('[Randomness]', seed, fix_seed))
print('{:12s} record_in_file={}'.format('[System]', record_in_file))
print('-------------------------------------------')

log('[Start Running]')
_, loss_path, acc_path, consensus_error_path = env.run()

record = {
    'dataset': data_package.name,
    'dataset_size': len(data_package.train_set),
    'dataset_feature_dimension': data_package.feature_dimension,
    'lr': super_params['lr'],
    'weight_decay': task.weight_decay,
    'honest_size': graph.honest_size,
    'byzantine_size': graph.byzantine_size,
    'rounds': env.rounds,
    'display_interval': env.display_interval,
    'total_iterations': env.total_iterations,
    'loss_path': loss_path,
    'acc_path': acc_path,
    'consensus_error_path': consensus_error_path,
    'fix_seed': fix_seed,
    'seed': seed,
    'graph': graph,
}

if record_in_file:
    path_list = [task.name, graph.name, env.partition_name] + workspace
    dump_file_in_cache(title, record, path_list=path_list)
print('-------------------------------------------')

