import os

seed = 100

gpu = '0'

attacks = [
   'label_flipping',
   'dynamic_label_flipping',
]

aggregations = [
   'mean',
   'ios',
   'trimmed-mean',
   'faba',
   'cc',
   'scc',
   'lfighter',
]
partitions = [
   'iid',
   'dirichlet_mild',
   'noniid'
]
graphs = [
   'two-castle',
   'fan',
   'line'
]

action = os.system
# action = print


# DSGD
for graph in graphs:
   for partition in partitions:
      for aggregation in aggregations:
         for attack in attacks:
            cmd = f'python "main DSGD-NN.py" ' \
               + f'--graph {graph} ' \
               + f'--seed {seed} '\
               + f'--aggregation {aggregation} ' \
               + f'--attack {attack} ' \
               + f'--data-partition {partition} '\
               + f'--gpu {gpu}'
            action(cmd)
