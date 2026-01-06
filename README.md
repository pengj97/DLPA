# Topology-Independent Robustness of the Weighted Mean under Label Poisoning Attacks in Heterogeneous Decentralized Learning

## Install
1. Download the dependant packages:
- python 3.9.18
- pytorch 1.13.1
- matplotlib 3.5.0
- networkx 2.6.3

2. Download the dataset to the directory `./dataset` and create a directory named `./record`. The experiment outputs will be stored in `./record`.

- *MNIST*: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- *CIFAR100*: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

## Construction
The main programs can be found in the following files:
- `ByrdLab`: main codes
- `main DSGD(-xxx).py`, : program entry
  * `main DSGD.py`: compute classification accuracies of different aggregators (Fig. 1, 3, 5, 6)
  * `main DSGD-hetero-disturb.py`: compute heterogeneity of regular gradients and disturbances of poisoned gradients (Fig. 4)
-  `draw_decentralized_multi_fig`: directories containing the codes that draw the figures in paper


## Runing
### Run DSGD
```bash
python "main DSGD.py"  --aggregation <aggregation-name> --attack <attack-name> --data-partition <data-partition> --graph <graph-name> --gpu <gpu-id> 
# ========================
# e.g.
# python "main DSGD.py" --aggregation mean --attack label_flipping --data-partition noniid --graph TwoCastle --gpu 0
```

> The arguments can be
>
>
> `<aggregation-name>`: 
> - mean
> - trimmed-mean
> - faba
> - cc
> - scc
> - ios
> - lfighter
>
> `<attack-name>`: 
> - label_flipping (which executes static label flipping attacks)
> - dynamic_label_flipping (which executes dynamic label flipping attacks)
>
> `<data-partition>`: 
> - iid
> - dirichlet_mild
> - noniid
>
> `<graph-name>`: 
> - TwoCastle
> - UnconnectedRegularLine
> - Fan
> - Lollipop

---


# ====================
# Figure 1
```
python run-toyexample.py

cd draw_decentralized_multi_fig

python draw-Decentral-toyexample.py

```

# ====================
# Figure 10
```
cd draw_decentralized_multi_fig

python draw-Decentral-topology.py
```

# ====================
# Figures 3, 6, 8
```
python run-experiments.py

cd draw_decentralized_multi_fig

python draw-Decentral-MultiFig.py
```

# ====================
# Figure 4
```
python run-hetero-disturb.py

cd draw_decentralized_multi_fig

python draw-Decentral-A-xi.py
```
