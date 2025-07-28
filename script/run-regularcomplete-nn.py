import os

graph = 'RegularComplete'

seed = 100

gpu = '0'

attacks = [
   'label_flipping',
   # 'label_random',
   # 'furthest_label_flipping',
]

aggregations = [
   'mean',
   'ios',
   'trimmed-mean',
   'faba',
   'cc',
   'scc',
   'lfighter'
]
partitions = [
   'iid',
   'dirichlet_mild',
   'noniid'
]

action = os.system
# action = print

# no communication
# for partition in partitions:
#    for attack in attacks:
#       cmd = f'python "main DSGD.py" ' \
#             + f'--graph {graph} ' \
#             + f'--aggregation no-comm ' \
#             + f'--attack {attack} ' \
#             + f'--data-partition {partition} ' \
#             + f'--gpu {gpu}'
#       action(cmd)

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

