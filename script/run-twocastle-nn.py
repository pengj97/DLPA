import os

# graph = 'Bipartite'
# graph = 'Complete'
# graph = 'LineGraph'
graph = 'TwoCastle'

seed = 100

gpu = '0'

attacks = [
   'label_flipping',
   # 'label_random',
   'furthest_label_flipping',
]

aggregations = [
   'mean',
   'ios',
   'trimmed-mean',
   'faba',
   'cc',
   'scc',
   'lfighter',
   # 'fldefender',
]
partitions = [
   'iid',
   'dirichlet_mild',
   'noniid'
]

action = os.system
# action = print

# Baseline (weighted mean without attacks)
# for partition in partitions:
#    cmd = f'python "main DSGD-NN.py" ' \
#          + f'--graph {graph} ' \
#          + f'--seed {seed} ' \
#          + f'--aggregation mean ' \
#          + f'--data-partition {partition} ' \
#          + f'--gpu {gpu}'
#    action(cmd)

# no communication
# for partition in partitions:
#       for attack in attacks:
#          cmd = f'python "main DSGD-NN.py" ' \
#                + f'--graph {graph} ' \
#                + f'--aggregation no-comm ' \
#                + f'--attack {attack} ' \
#                + f'--data-partition {partition} ' \
#                + f'--gpu {gpu}'
#          action(cmd)

# DSGD
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

