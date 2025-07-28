import argparse
import numpy as np
import random 
import torch
import os

import torch.backends
import torch.backends.cudnn 



parser = argparse.ArgumentParser(description='Robust Temporal Difference Learning')
    
# Arguments
parser.add_argument('--graph', type=str, default='Complete')
parser.add_argument('--aggregation', type=str, default='mean')
parser.add_argument('--attack', type=str, default='none')
parser.add_argument('--data-partition', type=str, default='iid')
# parser.add_argument('--lr-ctrl', type=str, default='1/sqrt k')
parser.add_argument('--lr-ctrl', type=str, default='constant')


parser.add_argument('--no-fixed-seed', action='store_true',
                    help="If specifed, the random seed won't be fixed")
parser.add_argument('--seed', type=int, default=200)

parser.add_argument('--without-record', action='store_true',
                    help='If specifed, no file of running record and log will be left')
parser.add_argument('--step-agg', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--prob', type=float, default=1.0)
parser.add_argument('--ER_prob', type=float, default=0.7)
parser.add_argument('--slack_param', type=float, default=1.0)


args = parser.parse_args()
gpu = args.gpu

if not args.no_fixed_seed:
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
