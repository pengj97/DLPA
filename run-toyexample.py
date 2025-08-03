import os

seed = 100

gpu = '0'

attacks = [
   'label_flipping',
]

aggregations = [
   # 'mean',
   # 'ios',
   # 'trimmed-mean',
   # 'faba',
   'cc',
   'scc',
]
partitions = [
   'noniid'
]

graphs = [
   'two-castle',
   'fan'
]

action = os.system
# action = print


# DSGD
for graph in graphs:
   for partition in partitions:
      for aggregation in aggregations:
         for attack in attacks:
            cmd = f'python "main DSGD.py" ' \
               + f'--graph {graph} ' \
               + f'--aggregation {aggregation} ' \
               + f'--attack {attack} ' \
               + f'--data-partition {partition} '\
               + f'--gpu {gpu}'
            action(cmd)
