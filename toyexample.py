import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# 1. Load dataset: Digits (10 classes) as example
digits = datasets.load_digits()
X = digits.data.astype(np.float64)
y = digits.target.astype(int)


# 2. Split into train/test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Standardize features
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

# 4. Non-IID Data partition: each node gets one class
def partition_data(X, y, num_nodes, partition_param):
    if partition_param == 'iid':
        indexs = list(range(len(X)))
        np.random.shuffle(indexs)
        sep = [(i * len(X)) // num_nodes for i in range(num_nodes + 1)]
        partition_index = [[indexs[i] for i in range(sep[node], sep[node+1])]
                                        for node in range(num_nodes)]
        partitions = [(X[partition_index[node]], y[partition_index[node]]) for node in range(num_nodes)]
    
    elif partition_param == 'mild_noniid':
        np.random.seed(100)

        alpha = 0.1
        classes = np.unique(y)
        num_classes = len(classes) 
        data_size = len(X)
        min_size = 10

        current_min_size = 0
        all_index = [[] for _ in range(num_classes)]
        for i, (_, label) in enumerate(zip(X, y)):
            all_index[label].append(i)

        partition_index = [[] for _ in range(num_nodes)]
        while current_min_size < min_size:
             partition_index = [[] for _ in range(num_nodes)]
             for k in range(num_classes):
                 idx_k = all_index[k]
                 np.random.shuffle(idx_k)
                 proportions = np.random.dirichlet(np.repeat(alpha, num_nodes))
                 # using the proportions from dirichlet, only select those nodes having data amount less than average
                 proportions = np.array(
                     [p * (len(idx_j) < data_size / num_nodes) for p, idx_j in zip(proportions, partition_index)]
                 )
                 proportions = proportions / proportions.sum()
                 proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                 partition_index = [idx_j + idx.tolist() for idx_j, idx in zip(partition_index, np.split(idx_k, proportions))]
                 current_min_size = min([len(idx_j) for idx_j in partition_index])
                 partitions = [(X[partition_index[node]], y[partition_index[node]]) for node in range(num_nodes)] 
    
    elif partition_param == 'noniid':
        classes = np.unique(y)
        num_classes = len(classes)
        assert num_nodes == num_classes, f"num_nodes ({num_nodes}) must equal number of classes ({num_classes})"
        partitions = []
        for i in range(num_nodes):
            Xi = X[y == i]
            yi = y[y == i]
            idx = np.arange(len(yi))
            np.random.shuffle(idx)
            partitions.append((Xi[idx], yi[idx]))
    
    return partitions

# 5. Define Linear SVM with gradient computation (Crammer-Singer)
class LinearSVM_GD:
    def __init__(self, num_features, num_classes, reg_strength=1e-4):
        self.W = np.zeros((num_features, num_classes), dtype=np.float64)
        self.b = np.zeros(num_classes, dtype=np.float64)
        self.reg_strength = reg_strength

    def compute_loss_and_grad(self, X, y):
        N, D = X.shape
        scores = X.dot(self.W) + self.b  # (N, C)
        correct_scores = scores[np.arange(N), y].reshape(-1, 1)
        margins = scores - correct_scores + 1.0
        margins[np.arange(N), y] = 0
        hinge = np.maximum(0, margins)
        loss = np.sum(hinge) / N + 0.5 * self.reg_strength * np.sum(self.W * self.W)
        mask = (hinge > 0).astype(np.float64)
        row_sum = np.sum(mask, axis=1)
        dscores = mask
        dscores[np.arange(N), y] = -row_sum
        dW = X.T.dot(dscores) / N + self.reg_strength * self.W
        db = np.sum(dscores, axis=0) / N
        return loss, dW, db

    def predict(self, X):
        scores = X.dot(self.W) + self.b
        return np.argmax(scores, axis=1)

# 6. Create lollipop and fan graph generators
def create_graph(W, P, topology):
    if topology == 'Complete':
        G, honest_nodes, byzantine_nodes = create_complete_graph(W, P)
    elif topology == 'Lollipop':
        G, honest_nodes, byzantine_nodes = create_lollipop_graph(W, P)
    elif topology == 'Fan':
        G, honest_nodes, byzantine_nodes = create_fan_graph(W, P)
    return G, honest_nodes, byzantine_nodes


def create_complete_graph(W, P):
    assert W > P, "Total nodes W must be > P"
    G = nx.complete_graph(W)
    honest_size = W - P
    honest_nodes = list(range(honest_size))
    byzantine_nodes = list(range(honest_size, W))

    return G, honest_nodes, byzantine_nodes

def create_lollipop_graph(W, P):
    assert W > P, "Total nodes W must be > P"
    honest_size = W - P
    honest_nodes = list(range(honest_size))
    byzantine_nodes = list(range(honest_size, W))
    G = nx.complete_graph(honest_size)
    # Attach byzantine nodes
    if honest_size >= 2 * P:
        attach_indices = honest_nodes[honest_size - 2*P: honest_size - P]
        if len(attach_indices) < P:
            attach_indices = honest_nodes[-P:]
    else:
        attach_indices = honest_nodes[-P:]
    if len(attach_indices) < P:
        attach_indices = (attach_indices + honest_nodes[-1:] * P)[:P]
    elif len(attach_indices) > P:
        attach_indices = attach_indices[-P:]
    for i, j in zip(attach_indices, byzantine_nodes):
        G.add_edge(i, j)
    for node in byzantine_nodes:
        if node not in G:
            G.add_node(node)
    return G, honest_nodes, byzantine_nodes

def create_fan_graph(W, P):
    assert W > P, "Total nodes W must be > P"
    honest_size = W - P
    honest_nodes = list(range(honest_size))
    byzantine_nodes = list(range(honest_size, W))
    G = nx.path_graph(honest_size)
    for b in byzantine_nodes:
        G.add_node(b)
        for i in honest_nodes:
            G.add_edge(i, b)
    return G, honest_nodes, byzantine_nodes



# 7. Compute Metropolis mixing matrix
def compute_mixing_matrix(G):
    num = G.number_of_nodes()
    E = np.zeros((num, num), dtype=np.float64)
    degrees = dict(G.degree())
    for i in G.nodes():
        for j in G.neighbors(i):
            E[i, j] = 1.0 / (1 + max(degrees[i], degrees[j]))
        E[i, i] = 1.0 - np.sum(E[i, :])
    return E

# Aggregaors
def WeiMean(E, local_models, node, byzantine_nodes):
    orig_shape = local_models[0].shape
    new_local_models = local_models.reshape(local_models.shape[0], -1)
    new_local_model = np.dot(E[node], new_local_models)
    new_local_model = new_local_model.reshape(orig_shape)
    return new_local_model


def neighbors_and_byzantine_size(E, node, byzantine_nodes):
    neighbors_and_itself = []
    neighbor_byzantine_size = 0
    for i in range(E.shape[0]):
        if E[node][i] > 0:
            neighbors_and_itself.append(i)
            if i in byzantine_nodes:
                neighbor_byzantine_size += 1
    
    # neighbors_and_itself = np.array(neighbors_and_itself)
    return neighbors_and_itself, neighbor_byzantine_size


def TriMean(E, local_models, node, byzantine_nodes):
    
    orig_shape = local_models[0].shape
    neighbors_and_itself, neighbor_byzantine_size = neighbors_and_byzantine_size(E, node, byzantine_nodes)
    neighbors_and_itself_models = local_models[neighbors_and_itself]
    new_local_models = neighbors_and_itself_models.reshape(neighbors_and_itself_models.shape[0], -1)


    # 将张量按列排序
    sorted_models = np.sort(new_local_models, axis=0)
    if neighbor_byzantine_size == 0:
        trimmed_data = sorted_models
    elif neighbor_byzantine_size > 0:
        trimmed_data = sorted_models[neighbor_byzantine_size:-neighbor_byzantine_size, :]
    else: 
        assert False, 'Poisoning agent size should be equal or larger than 0!'

    # 计算修剪后的均值
    if len(trimmed_data) > 0:
        tm = np.mean(trimmed_data, axis=0)
        
    else:
        tm = np.zeros(orig_shape)
    
    return tm.reshape(orig_shape)


def FABA(E, local_models, node, byzantine_nodes):
    orig_shape = local_models[0].shape
    neighbors_and_itself, neighbor_byzantine_size = neighbors_and_byzantine_size(E, node, byzantine_nodes)
    neighbors_and_itself_models = local_models[neighbors_and_itself]
    new_local_models = neighbors_and_itself_models.reshape(neighbors_and_itself_models.shape[0], -1)

    remain = new_local_models
    for _ in range(neighbor_byzantine_size):
        mean = np.mean(remain, axis=0)
        # remove the largest 'byzantine_size' model
        distances = np.array([
            np.linalg.norm(model - mean) for model in remain
        ])
        remove_index = distances.argmax()
        remain = remain[np.arange(remain.shape[0]) != remove_index]
    
    if len(remain) == 0:
        result = np.zeros(orig_shape)
    else:
        result = np.mean(remain, axis=0)
    
    return result.reshape(orig_shape)


def IOS(E, local_models, node, byzantine_nodes):
    orig_shape = local_models[0].shape
    neighbors_and_itself, neighbor_byzantine_size = neighbors_and_byzantine_size(E, node, byzantine_nodes)
    neighbors_and_itself_models = local_models[neighbors_and_itself]
    new_local_models = neighbors_and_itself_models.reshape(neighbors_and_itself_models.shape[0], -1)

    remain_models = new_local_models
    remain_weight = E[node][neighbors_and_itself]
    for _ in range(neighbor_byzantine_size):
        mean = np.dot(remain_weight, remain_models)
        mean /= np.sum(remain_weight)
        distances = np.array([
            np.linalg.norm(model - mean) for model in remain_models
        ])
        remove_idx = distances.argmax()
        remain_idx = np.arange(remain_models.shape[0]) != remove_idx
        remain_models = remain_models[remain_idx]
        remain_weight = remain_weight[remain_idx]

    result = np.dot(remain_weight, remain_models)
    result /= np.sum(remain_weight)

    return result.reshape(orig_shape)


def CC(E, local_models, node, byzantine_nodes):
    threshold = 0.03
    orig_shape = local_models[0].shape
    neighbors_and_itself, neighbor_byzantine_size = neighbors_and_byzantine_size(E, node, byzantine_nodes)
    neighbors_and_itself_models = local_models[neighbors_and_itself]
    new_local_models = neighbors_and_itself_models.reshape(neighbors_and_itself_models.shape[0], -1)

    local_model = local_models[node].copy()
    v0 = local_model.reshape(-1)
    diff = np.zeros_like(v0)
    for model in new_local_models:
        norm = np.linalg.norm(model - v0)
        if norm > threshold:
            diff += threshold * (model - v0) / norm
        else:
            diff += model - v0
    
    diff /= len(neighbors_and_itself)
    v0 = v0 + diff
    return v0.reshape(orig_shape)


def CG(E, local_models, node, byzantine_nodes):
    threshold = 0.03
    orig_shape = local_models[0].shape
    neighbors_and_itself, neighbor_byzantine_size = neighbors_and_byzantine_size(E, node, byzantine_nodes)

    local_model = local_models[node].copy()
    local_model = local_model.reshape(-1)
    diff = np.zeros_like(local_model)
    for i in neighbors_and_itself:
        model = local_models[i].reshape(-1)
        norm = np.linalg.norm(model - local_model)
        weight = E[node][i]
        if norm > threshold:
            diff += weight * threshold * (model - local_model) / norm
        else:
            diff += weight * (model - local_model)
    
    
    local_model = local_model + diff
    return local_model.reshape(orig_shape)


# Define label poisoning attacks
def label_flipping(X, y, num_classes, Ws, bs):
    y_use = num_classes - 1 -y
    return y_use


def furthest_label_flipping(X, y, num_classes, Ws, bs):
    W_avg = np.mean(np.stack(Ws, axis=0), axis=0)
    b_avg = np.mean(np.stack(bs, axis=0), axis=0)
    y_use = np.argmin(X.dot(W_avg) + b_avg, axis=1)
    return y_use



# 8. Decentralized training: 先梯度下降，再聚合；包含 label flipping 攻击
def decentralized_train_label_flipping_grad_then_agg(partitions, E, byzantine_nodes, 
                                                     aggregation, attacks,
                                                     num_features=64, num_classes=10,
                                                     reg_strength=1e-4, num_iters=500,
                                                     learning_rate=1e-2, verbose=False):
    num_nodes = len(partitions) 
    # Initialize local parameters
    Ws = [np.zeros((num_features, num_classes), dtype=np.float64) for _ in range(num_nodes)]
    bs = [np.zeros(num_classes, dtype=np.float64) for _ in range(num_nodes)]
    model = LinearSVM_GD(num_features, num_classes, reg_strength=reg_strength)
    loss_history = []
    train_acc_history = []
    test_acc_history = []

    for it in range(num_iters):
        # 1) 本地梯度下降步：计算 x^{k+1/2} = x^k - lr * grad
        W_half = np.zeros_like(Ws)
        b_half = np.zeros_like(bs)
        losses = [0.0] * num_nodes
        lr = learning_rate / np.sqrt(it+1)
        for i in range(num_nodes):
            X_part, y_part = partitions[i]
            print(f'Node {i} training with {len(X_part)} samples')
            # 恶意节点执行 label flipping: b -> (num_classes - 1 - b)
            if i in byzantine_nodes:
                # y_use = (num_classes - 1) - y_part
                y_use = attacks(X_part, y_part, num_classes, Ws, bs)
            else:
                y_use = y_part
            # 设置当前模型参数
            model.W = Ws[i].copy()
            model.b = bs[i].copy()
            # 计算本地损失和梯度
            loss_i, grad_W_i, grad_b_i = model.compute_loss_and_grad(X_part, y_use)
            # 梯度下降得到中间参数
            W_half[i] = Ws[i] - lr * grad_W_i
            b_half[i] = bs[i] - lr * grad_b_i
            losses[i] = loss_i
        # 2) 聚合中间参数：x^{k+1} = Agg(x_j^{k+1/2})
        for i in range(num_nodes):
            Ws[i] = aggregation(E, W_half, i, byzantine_nodes)
            bs[i] = aggregation(E, b_half, i, byzantine_nodes)

        W_avg = np.mean(np.stack(Ws, axis=0), axis=0)
        b_avg = np.mean(np.stack(bs, axis=0), axis=0)
        y_train_pred = np.argmax(X_train_full.dot(W_avg) + b_avg, axis=1)
        y_test_pred = np.argmax(X_test.dot(W_avg) + b_avg, axis=1)
        train_acc = accuracy_score(y_train_full, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        avg_loss = np.mean(losses)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        loss_history.append(avg_loss)
        
        if verbose and (it % 50 == 0 or it == num_iters - 1):
            print(f"{aggregation.__name__}: Iteration {it+1}/{num_iters}, train acc: {train_acc:.4f} test acc: {test_acc:.4f} avg local loss: {avg_loss:.4f}")
            
    return train_acc_history, test_acc_history, loss_history

# === Example usage ===
W = 10
P = 1
num_features = X_train_full.shape[1]
num_classes = 10  # 10 classes

# topologies = ['Complete', 'Lollipop', 'Fan']
topologies = ['Lollipop', 'Fan']
# topologies = ['Complete', 'Fan']
# topologies = ['Lollipop']
aggregations = [ TriMean, FABA, CC, CG,  IOS, WeiMean]
# aggregations = [FABA]
attacks = label_flipping
# attacks = furthest_label_flipping
# partition_param = 'iid'
# partition_param = 'mild_noniid'
partition_param = 'noniid'
results = {}
num_iters = 1000
learning_rate = 1e-2

for topology in topologies:
    for aggregation in aggregations:
        G, honest_nodes, byzantine_nodes = create_graph(W, P, topology)  

        # Non-IID: each节点只含一种数字类别样本
        partitions = partition_data(X_train_full, y_train_full, W, partition_param)

        # 计算混合矩阵
        E = compute_mixing_matrix(G)

        # 去中心化训练：先梯度下降再聚合，包含标签翻转攻击
        train_acc_history, test_acc_history, loss_history = decentralized_train_label_flipping_grad_then_agg(
            partitions, E, byzantine_nodes, aggregation, attacks, num_features, num_classes,
            reg_strength=1e-4, num_iters=num_iters, learning_rate=learning_rate, verbose=True
        )

        results[topology] = {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'test_acc_history': test_acc_history,
            'G': G,
            'honest_nodes': honest_nodes,
            'byzantine_nodes': byzantine_nodes
        }

        file_path = f'record/SVM_digits/{topology}_n={W}_b={P}/{partition_param}/DSGD_{attacks.__name__}_{aggregation.__name__}_invSqrtLR'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(results[topology], f)

# 9. 绘制与可视化
FONTSIZE = 25
colors = [ 'orange', 'blue', 'purple', 'grey', 'gold',  'red']
markers = [ 'v',  's', 'x', 'o', '>',  'D']

# 先绘制网络图部分
for i in range(len(topologies)):
    file_path = f'record/SVM_digits/{topologies[i]}_n={W}_b={P}/noniid/DSGD_label_flipping_WeiMean_invSqrtLR'
    with open(file_path, 'rb') as f:
        record = pickle.load(f)
    G = record['G']
    honest_nodes = record['honest_nodes']
    byzantine_nodes = record['byzantine_nodes']
    fig = plt.figure(figsize=(6, 6))
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True,
            node_color=['#99CCCC' if i in honest_nodes else '#FF6666' for i in G.nodes()],
            edge_color='gray', ax=fig.add_subplot(111))
    plt.title(f"{topologies[i].capitalize()} Topology (W={W}, P={P})")
    plt.show()

# 然后创建新的图形用于绘制准确率曲线
fig, axes = plt.subplots(1, len(topologies), figsize=(7 * len(topologies), 8), sharex=True, sharey=True)

axes[0].set_ylabel('Accuracy', fontsize=FONTSIZE)
axes[0].set_ylim(0.65, 0.87)

taskname = 'SVM_digits'
for i in range(len(topologies)):
    axes[i].set_title(f'{topologies[i]} graph', fontsize=FONTSIZE)
    axes[i].set_xlabel('iterations', fontsize=FONTSIZE)
    axes[i].tick_params(labelsize=FONTSIZE)
    axes[i].grid(True)  # 使用True而不是'on'
    for index in range(len(aggregations)):
        color = colors[index]
        marker = markers[index]
        file_path = f'record/SVM_digits/{topologies[i]}_n={W}_b={P}/{partition_param}/DSGD_{attacks.__name__}_{aggregations[index].__name__}_invSqrtLR'
        with open(file_path, 'rb') as f:
            record = pickle.load(f)
        test_acc_history = record['test_acc_history']
        x_axis = [r for r in range(num_iters)]
        axes[i].plot(x_axis, test_acc_history, '-', color=color, marker=marker, 
                    label=aggregations[index].__name__, markevery=100, linewidth=4, markersize=10)

handles, labels = axes[0].get_legend_handles_labels() 
leg = fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=FONTSIZE, markerscale=2)
leg_lines = leg.get_lines()
for i in range(len(leg_lines)):
    plt.setp(leg_lines[i], linewidth=5.0)   

plt.subplots_adjust(top=0.91,
                   bottom=0.29,
                   left=0.1,
                   right=0.97,
                   hspace=0.27,
                   wspace=0.15)

plt.savefig(f'toyexample_{partition_param}_{attacks.__name__}_multiclass.pdf', format='pdf', bbox_inches='tight')
plt.show()