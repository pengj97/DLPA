import os

graph = 'two-castle'

seed = 100

gpu = '2'

attacks = [
   'label_flipping',
]

partitions = [
   'iid',
   'dirichlet_mild',
   'noniid'
]

action = os.system


# DSGD
for partition in partitions:
      for attack in attacks:
         cmd = f'python "main DSGD-hetero-disturb.py" ' \
         + '--aggregation mean ' \
         + f'--graph {graph} ' \
         + f'--seed {seed} '\
         + f'--attack {attack} ' \
         + f'--data-partition {partition} '\
         + f'--gpu {gpu}'
         action(cmd)
         
