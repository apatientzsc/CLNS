import collections

import torch
import torch.nn.functional as F
import numpy as np


def new_generate_correlations(features, pseudo_labels, cluster_nums, r, coe):
    features = F.normalize(features, dim=1)
    sim = features.mm(features.t())
    assert sim.shape[0] == pseudo_labels.shape[0]

    # 对于每个样本，找到最高相似度值
    max_sim_values, _ = torch.max(sim, dim=1, keepdim=True)

    # 计算阈值：r * 最高相似度值
    thresholds = r * max_sim_values

    # 找到所有大于阈值的邻居
    neighbor_masks = sim >= thresholds

    correlations_all = []
    weights_all = []

    cluster_nums = torch.tensor(cluster_nums)

    for i in range(sim.shape[0]):
        correlation = torch.zeros(cluster_nums.shape[0])
        neighbor_indices = torch.where(neighbor_masks[i])[0]
        neighbor_count = 0

        for j in neighbor_indices:
            pid = pseudo_labels[j.item()]
            if pid.item() != -1:
                correlation[pid.item()] += 1
            neighbor_count += 1

        correlation = correlation / (neighbor_count - correlation + cluster_nums)

        correlation /= correlation.sum()
        correlations_all.append(correlation)

        if pseudo_labels[i].item() == -1:
            weights_all.append(0)
        else:

            weight = torch.exp(
                -coe * (1.0 - correlation[pseudo_labels[i].item()]) * (1.0 - correlation[pseudo_labels[i].item()]))
            weights_all.append(weight)

    correlations_all = torch.stack(correlations_all, dim=0)
    weights_all = torch.tensor(weights_all)
    return correlations_all, weights_all


def generate_correlations(features, pseudo_labels, cluster_nums, neighbour_num, coe):
    features = F.normalize(features, dim=1)
    sim = features.mm(features.t())
    assert sim.shape[0] == pseudo_labels.shape[0]
    _, inds = torch.topk(sim, neighbour_num, dim=1, largest=True, sorted=True)
    correlations_all = []
    weights_all = []

    cluster_nums = torch.tensor(cluster_nums)


    for i in range(inds.shape[0]):
        correlation = torch.zeros(cluster_nums.shape[0])
        for j in range(inds.shape[1]):
            pid = pseudo_labels[inds[i, j].item()]
            if pid.item() != -1:
                correlation[pid.item()] += 1


        correlation = correlation / (neighbour_num - correlation + cluster_nums)

        correlation /= correlation.sum()
        correlations_all.append(correlation)
        #weight = torch.log(correlation[pseudo_labels[i].item()]+1.0)
        if pseudo_labels[i].item() == -1:
            weights_all.append(0)
        else:
            weight = torch.exp(-coe * (1.0 - correlation[pseudo_labels[i].item()]) * (1.0 - correlation[pseudo_labels[i].item()]))
            weights_all.append(weight)
    correlations_all = torch.stack(correlations_all, dim=0)
    weights_all = torch.tensor(weights_all)
    return correlations_all, weights_all


def new_generate_correlations_cross(features, features_cross, pseudo_labels, cluster_nums, r, pseudo_labels_own,
                                pid_transform, coe):
    features = F.normalize(features, dim=1)
    features_cross = F.normalize(features_cross, dim=1)
    sim = features.mm(features_cross.t())
    assert sim.shape[1] == pseudo_labels.shape[0]
    assert sim.shape[0] == pseudo_labels_own.shape[0]

    # 对于每个样本，找到最高相似度值
    max_sim_values, _ = torch.max(sim, dim=1, keepdim=True)

    # 计算阈值：r * 最高相似度值
    thresholds = r * max_sim_values

    # 找到所有大于阈值的邻居
    neighbor_masks = sim >= thresholds

    correlations_all = []
    weights_all = []

    cluster_nums = torch.tensor(cluster_nums)

    for i in range(sim.shape[0]):
        correlation = torch.zeros(cluster_nums.shape[0])
        neighbor_indices = torch.where(neighbor_masks[i])[0]
        neighbor_count = 0

        for j in neighbor_indices:
            pid = pseudo_labels[j.item()]
            if pid.item() != -1:
                correlation[pid.item()] += 1
            neighbor_count += 1

        correlation = correlation / (neighbor_count - correlation + cluster_nums)

        correlation /= correlation.sum()
        correlations_all.append(correlation)

        if pseudo_labels_own[i].item() == -1:
            weights_all.append(0)
        else:
            weight = torch.exp(-coe * (1.0 - correlation[pid_transform[pseudo_labels_own[i].item()]]) * (
                        1.0 - correlation[pid_transform[pseudo_labels_own[i].item()]]))
            weights_all.append(weight)

    correlations_all = torch.stack(correlations_all, dim=0)
    weights_all = torch.tensor(weights_all)
    return correlations_all, weights_all

def generate_correlations_cross(features, features_cross, pseudo_labels, cluster_nums, neighbour_num, pseudo_labels_own, pid_transform, coe):
    features = F.normalize(features, dim=1)
    features_cross = F.normalize(features_cross, dim=1)
    sim = features.mm(features_cross.t())
    assert sim.shape[1] == pseudo_labels.shape[0]
    assert sim.shape[0] == pseudo_labels_own.shape[0]
    _, inds = torch.topk(sim, neighbour_num, dim=1, largest=True, sorted=True)
    correlations_all = []
    weights_all = []

    cluster_nums = torch.tensor(cluster_nums)

    for i in range(inds.shape[0]):
        correlation = torch.zeros(cluster_nums.shape[0])
        for j in range(inds.shape[1]):
            pid = pseudo_labels[inds[i, j].item()]
            if pid.item() != -1:
                correlation[pid.item()] += 1

        correlation = correlation / (neighbour_num - correlation + cluster_nums)

        correlation /= correlation.sum()
        correlations_all.append(correlation)
        if pseudo_labels_own[i].item() == -1:
            weights_all.append(0)
        else:
            weight = torch.exp(-coe * (1.0 - correlation[pid_transform[pseudo_labels_own[i].item()]]) * (1.0 - correlation[pid_transform[pseudo_labels_own[i].item()]]))
            weights_all.append(weight)
    correlations_all = torch.stack(correlations_all, dim=0)
    weights_all = torch.tensor(weights_all)
    return correlations_all, weights_all


def fake_generate_correlations_cross(features, features_cross, pseudo_labels, cluster_nums, neighbour_num, pseudo_labels_own,
                                pid_transform, coe):
    # 生成与原始输出形状相同的随机数据
    batch_size = features.shape[0]
    num_clusters = len(cluster_nums)

    # 随机生成correlations_all (形状: [batch_size, num_clusters])
    correlations_all = torch.rand(batch_size, num_clusters)
    correlations_all = correlations_all / correlations_all.sum(dim=1, keepdim=True)  # 归一化

    # 随机生成weights_all (形状: [batch_size])
    weights_all = torch.rand(batch_size)


    return correlations_all, weights_all



def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    cluster_nums = [len(centers[idx]) for idx in sorted(centers.keys())]

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers, cluster_nums