import os

graph = 'TwoCastle'

# seeds = list(range(0, 1000, 100))
seeds = [100]

gpu = '3'

attacks = [
   'label_flipping',
   # 'furthest_label_flipping',
   # 'label_random',
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
         for seed in seeds:
            cmd = f'python "main DSGD-hetero-disturb.py" ' \
            + f'--graph {graph} ' \
            + f'--seed {seed} '\
            + f'--attack {attack} ' \
            + f'--data-partition {partition} '\
            + f'--gpu {gpu}'
            action(cmd)
         
