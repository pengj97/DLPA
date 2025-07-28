from argsParser import args

from ByrdLab import FEATURE_TYPE
from ByrdLab.aggregation import (D_bulyan, D_faba, D_geometric_median, D_Krum, 
                                 D_ios_equal_neigbor_weight, D_trimmed_mean,
                                 D_meanW, D_median, D_no_communication,
                                 D_remove_outliers, D_self_centered_clipping,
                                 D_mKrum, D_centered_clipping, D_ios, D_brute, D_mean, D_LFighter, D_FLGT, D_FLDefender)
from ByrdLab.attack import (D_alie, D_gaussian, D_isolation_weight,
                            D_sample_duplicate, D_sign_flipping,
                            D_zero_sum, D_zero_value, label_flipping, label_random, feature_label_random, furthest_label_flipping, adversarial_label_flipping)
from ByrdLab.decentralizedAlgorithm import DSGD, DSGD_under_DPA, DSGD_MSG, DSGD_MSG_under_DPA, DSGD_with_LFighter_under_DPA
from ByrdLab.graph import CompleteGraph, ErdosRenyi, OctopusGraph, TwoCastle, LineGraph, RandomGeometricGraph, RingGraph,\
                          Graph_byz_nodes_on_shortest_paths, Graph_byz_nodes_Not_on_shortest_paths, StarGraph, RegularCompleteGraph, TwoHeadLineGraph, FanGraph, UnconnectedRegularLineGraph
from ByrdLab.library.cache_io import dump_file_in_cache, load_file_in_cache
from ByrdLab.library.dataset import ijcnn, mnist, fashionmnist, cifar10, mnist_sorted_by_labels, cifar100
from ByrdLab.library.learnRateController import ladder_lr, one_over_sqrt_k_lr
from ByrdLab.library.partition import (LabelSeperation, TrivalPartition,
                                   iidPartition, DirichletMildPartition)
from ByrdLab.library.tool import log
from ByrdLab.tasks.logisticRegression import LogisticRegressionTask
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask
from ByrdLab.tasks.leastSquare import LeastSquareToySet, LeastSquareToyTask
from ByrdLab.tasks.neuralNetwork import NeuralNetworkTask
from ByrdLab.tasks.multiLayerPerceptron import MultiLayerPerceptronTask
from ByrdLab.tasks.ResNet import ResNetTask

# args.graph = 'TwoCastle'
# args.graph = 'Fan'
# args.ER_prob = '0.2'
# args.graph = 'Star'
# args.graph = 'byz_not_on_shortest'
# args.attack = 'furthest_label_flipping'
# args.attack = 'label_flipping'
# args.attack = 'sign_flipping'
# args.attack = 'sample_duplicate'
# args.lr_ctrl = 'constant'
# args.lr_ctrl = 'ladder'
args.lr_ctrl = '1/sqrt k'
# args.data_partition = 'noniid'
# args.data_partition = 'dirichlet_mild'
# args.data_partition = 'iid'
# args.aggregation = 'lfighter' 
# args.aggregation = 'ios' 
# args.aggregation = 'scc' 
# args.gpu = 2

# run for decentralized algorithm
# -------------------------------------------
# define graph
# -------------------------------------------
if args.graph == 'Complete':
    graph = CompleteGraph(node_size=10, byzantine_size=0)
elif args.graph == 'RegularComplete':
    graph = RegularCompleteGraph(node_size=10, byzantine_size=1)
elif args.graph == 'TwoCastle':
    graph = TwoCastle(k=5, byzantine_size=1, seed=40) # generate graph with 2k nodes
elif args.graph == 'ER':
    honest_size = 9
    byzantine_size = 1
    node_size = honest_size + byzantine_size
    # graph = ErdosRenyi(node_size, byzantine_size, float(args.ER_prob), seed=300)
    graph = ErdosRenyi(node_size, byzantine_size, float(args.ER_prob), seed=args.seed)
elif args.graph == 'RGG':
    graph = RandomGeometricGraph(node_size=14, byzantine_size=4, radius=1, seed=300)
elif args.graph == 'Octopus':
    graph = OctopusGraph(5, 0, 5)
elif args.graph == 'Line':
    honest_size = 10
    byzantine_size = 0
    node_size = honest_size + byzantine_size
    graph = LineGraph(node_size=node_size, byzantine_size=byzantine_size)
elif args.graph == 'TwoHeadLine':
    honest_size = 8
    byzantine_size = 2
    node_size = honest_size + byzantine_size
    graph = TwoHeadLineGraph(node_size=node_size, byzantine_size=byzantine_size)
elif args.graph == 'Ring':
    honest_size = 9
    byzantine_size = 1
    node_size = honest_size + byzantine_size
    graph = RingGraph(node_size=node_size, byzantine_size=byzantine_size)
elif args.graph == 'Star':
    honest_size = 9
    byzantine_size = 1
    node_size = honest_size + byzantine_size
    graph = StarGraph(node_size=node_size, byzantine_size=byzantine_size)
elif args.graph == 'byz_on_shortest':
    node_size = 4
    graph = Graph_byz_nodes_on_shortest_paths(node_size=node_size)
elif args.graph == 'byz_not_on_shortest':
    node_size = 4
    graph = Graph_byz_nodes_Not_on_shortest_paths(node_size=node_size)
elif args.graph == 'Fan':
    honest_size = 10
    byzantine_size = 0
    node_size = honest_size + byzantine_size
    graph = FanGraph(node_size=node_size, byzantine_size=byzantine_size)
elif args.graph == 'Fan2':
    honest_size = 9
    byzantine_size = 1
    node_size = honest_size + byzantine_size
    graph = FanGraph(node_size=node_size, byzantine_size=byzantine_size)
elif args.graph == 'UnconnectedRegularLine':
    graph = UnconnectedRegularLineGraph(node_size=10, byzantine_size=1)
else:
    assert False, 'unknown graph'
    
# If the Byzantine nodes present, they will do the attack even they do not change their model 
# over time. The presence of Byzantine nodes is an attack.
if args.attack == 'none':
    graph = graph.honest_subgraph()

# ===========================================

# -------------------------------------------
# define learning task
# -------------------------------------------
# data_package = ijcnn()
# task = LogisticRegressionTask(data_package)

# dataset = ToySet(set_size=500, dimension=5, fix_seed=True)

# data_package = mnist()
# task = softmaxRegressionTask(data_package, batch_size=32)

# data_package = fashionmnist()
# task = softmaxRegressionTask(data_package)

# data_package = fashionmnist()
# task = MultiLayerPerceptronTask(data_package, batch_size=32)

# data_package = cifar10()
# task = NeuralNetworkTask(data_package, batch_size=32)

# data_package = mnist()
# task = MultiLayerPerceptronTask(data_package, batch_size=32)

data_package = cifar100()
task = ResNetTask(data_package, batch_size=32)

# data_package = cifar10()
# task = ResNetTask(data_package, batch_size=32)

# w_star = torch.tensor([1], dtype=FEATURE_TYPE)
# data_package = LeastSquareToySet(set_size=2000, dimension=1, w_star=w_star, noise=0, fix_seed=True)
# data_package = LeastSquareToySet(set_size=100, dimension=1, noise=0, fix_seed=True)
# task = LeastSquareToyTask(data_package)

# task.super_params['display_interval'] = 20000
# task.super_params['rounds'] = 10
# task.super_params['lr'] = 0.004
# task.super_params['lr'] = 0.1
# task.initialize_fn = None
# ===========================================


# -------------------------------------------
# define learning rate control rule
# -------------------------------------------
if args.lr_ctrl == 'constant':
    lr_ctrl = None
elif args.lr_ctrl == '1/sqrt k':
    lr_ctrl = one_over_sqrt_k_lr(a=1, b=1)
    # super_params = task.super_params
    # total_iterations = super_params['rounds']*super_params['display_interval']
    # lr_ctrl = one_over_sqrt_k_lr(total_iteration=total_iterations,
    #                              a=math.sqrt(1001), b=1000)
elif args.lr_ctrl == 'ladder':
    # decreasing_iter_ls = [5000, 10000, 15000]
    # proportion_ls = [0.3, 0.2, 0.1]
    decreasing_iter_ls = [100, 200, 300, 400, 500, 600, 700, 800, 900]  
    proportion_ls = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # decreasing_iter_ls = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  
    # proportion_ls = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
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
    # aggregation = D_mean(graph)
elif args.aggregation == 'ios':
    aggregation = D_ios(graph)
elif args.aggregation == 'ios_exact_byz_cnt':
    aggregation = D_ios(graph, exact_byz_cnt=False, byz_cnt=2)
elif args.aggregation == 'ios_equal_neigbor_weight':
    aggregation = D_ios_equal_neigbor_weight(graph)
elif args.aggregation == 'trimmed-mean':
    aggregation = D_trimmed_mean(graph)
elif args.aggregation == 'median':
    aggregation = D_median(graph)
elif args.aggregation == 'geometric-median':
    aggregation = D_geometric_median(graph)
elif args.aggregation == 'faba':
    aggregation = D_faba(graph)
elif args.aggregation == 'remove-outliers':
    aggregation = D_remove_outliers(graph)
elif args.aggregation == 'mKrum':
    aggregation = D_mKrum(graph)
elif args.aggregation == 'Krum':
    aggregation = D_Krum(graph)
elif args.aggregation == 'bulyan':
    aggregation = D_bulyan(graph)
elif args.aggregation == 'brute':
    aggregation = D_brute(graph)
elif args.aggregation == 'cc':
    if args.data_partition == 'iid':
        # threshold = 0.1
        # threshold = 0.01
        # threshold = 100
        # threshold = 90
        threshold = 100

    else:
        # threshold = 0.3
        # threshold = 0.03
        # threshold = 300
        threshold = 100

    aggregation = D_centered_clipping(graph, threshold=threshold)
elif args.aggregation == 'scc':
    if args.data_partition == 'iid':
        # threshold = 0.1
        # threshold = 0.01
        # threshold = 100
        threshold = 0.1

    else:
        # threshold = 0.3
        # threshold = 0.03
        # threshold = 300
        # threshold = 0.3
        threshold = 0.1

    aggregation = D_self_centered_clipping(
        graph, threshold_selection='parameter', threshold=threshold)
    # aggregation = D_self_centered_clipping(
    #     graph, threshold_selection='estimation')
elif args.aggregation == 'lfighter':
    aggregation = D_LFighter(graph)
elif args.aggregation == 'flgt':
    aggregation = D_FLGT(graph)
elif args.aggregation == 'fldefender':
    aggregation = D_FLDefender(graph)
else:
    assert False, 'unknown aggregation'

# ===========================================
    
# -------------------------------------------
# define attack
# -------------------------------------------
if args.attack == 'none':
    attack = None
elif args.attack == 'label_flipping':
    attack = label_flipping()
elif args.attack == 'label_random':
    attack = label_random()
elif args.attack == 'feature_label_random':
    attack = feature_label_random()
elif args.attack == 'furthest_label_flipping':
    attack = furthest_label_flipping()
elif args.attack == 'adversarial_label_flipping_iid':
    attack = adversarial_label_flipping()

    path = ['SR_mnist', 'Complete_n=1_b=0', 'TrivalPartition', 'best']
    q = load_file_in_cache('q-end', path_list=path)
    data_size = len(data_package.train_set)
    num_classes = data_package.num_classes
    len_q = num_classes * data_size
    assert len(q) == len_q

    for i in range(len_q):
        if q[i] == 1:
            k = i // data_size
            index = i % data_size
            task.data_package.train_set.targets[index] = (task.data_package.train_set.targets[index] + k) % num_classes

elif args.attack == 'adversarial_label_flipping_noniid':
    attack = adversarial_label_flipping()
    data_package = mnist_sorted_by_labels()
    task = softmaxRegressionTask(data_package)
    partition_cls = LabelSeperation

    path = ['SR_mnist', 'Complete_n=1_b=0', 'LabelSeperation', 'best']
    q = load_file_in_cache('q-end', path_list=path)
    ratio = graph.byzantine_size / graph.node_size
    flipped_data_size = int(ratio * len(data_package.train_set))
    num_classes = data_package.num_classes
    len_q = num_classes * flipped_data_size
    assert len(q) == len_q

    for i in range(len_q):
        if q[i] == 1:
            k = i // flipped_data_size
            index = i % flipped_data_size
            task.data_package.train_set.targets[index] = (task.data_package.train_set.targets[index] + k) % num_classes
elif args.attack == 'sign_flipping':
    attack = D_sign_flipping(graph)
elif args.attack == 'gaussian':
    attack = D_gaussian(graph)
elif args.attack == 'isolation':
    # D_isolation(graph),
    attack = D_isolation_weight(graph)
elif args.attack == 'sample_duplicate':
    attack = D_sample_duplicate(graph)
elif args.attack == 'zero_sum':
    attack = D_zero_sum(graph)
elif args.attack == 'zero_value':
    attack = D_zero_value(graph)
elif args.attack == 'alie':
    attack = D_alie(graph)
    # D_alie(graph, scale=2)
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
    if args.aggregation == 'lfighter' or args.aggregation == 'fldefender':
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
    # path_list = [task.name, f'{graph.name}_slack={args.slack_param}', env.partition_name] + workspace
    dump_file_in_cache(title, record, path_list=path_list)
print('-------------------------------------------')

