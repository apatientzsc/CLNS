import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional



def cam_prototype_calibration(features, pseudo_labels, cams, k_neighbors, confidence_threshold):
    """
    Prototype calibration with camera-aware neighbor filtering

    Args:
        features: [N, D] Feature vectors
        pseudo_labels: [N] Pseudo labels
        cams: [N] Camera IDs

    Returns:
        new_pseudo_labels: [N] Optimized pseudo labels
    """
    print(f"Using prototype calibration with cross-camera KNN filtering. K={k_neighbors}, Confidence threshold={confidence_threshold}")
    N = features.size(0)
    device = features.device

    # Normalize features
    features = F.normalize(features, p=2, dim=1)

    # Compute pairwise distances
    dists = torch.cdist(features, features, p=2)  # [N, N]

    # Create camera difference matrix: [N, N]
    cam_diff = (cams.unsqueeze(1) != cams.unsqueeze(0))  # True where cams are different

    # Set distances of same-camera samples to a large value to exclude them
    dists_same_cam = dists.clone()
    dists_same_cam[~cam_diff] = float('inf')  # Mask same-camera distances

    # Get K nearest cross-camera neighbors
    _, knn_idx = torch.topk(dists_same_cam, k=min(k_neighbors, (cam_diff.sum(dim=1)).min().item()), dim=1, largest=False)  # [N, k]

    # Compute confidence scores based on label agreement with cross-camera neighbors
    same_label = (pseudo_labels[knn_idx] == pseudo_labels.view(-1, 1))  # [N, k]
    confidence = same_label.float().mean(dim=1)  # [N]

    # Compute prototypes from high-confidence samples per class
    unique_labels = torch.unique(pseudo_labels)
    prototypes = {}
    for label in unique_labels:
        mask = (pseudo_labels == label) & (confidence >= confidence_threshold)
        if mask.sum() == 0:
            mask = (pseudo_labels == label)
        if mask.sum() == 0:
            continue
        prototypes[label.item()] = features[mask].mean(dim=0)  # [D]

    if not prototypes:
        return pseudo_labels  # Return original if no prototype found

    proto_labels = list(prototypes.keys())
    proto_feats = torch.stack([prototypes[l] for l in proto_labels], dim=0)  # [C, D]

    # Compute distances to prototypes
    dists_to_proto = torch.cdist(features, proto_feats, p=2)  # [N, C]

    # Assign pseudo label of nearest prototype
    nearest_proto_idx = torch.argmin(dists_to_proto, dim=1)
    new_pseudo_labels = torch.tensor([proto_labels[i] for i in nearest_proto_idx.cpu().tolist()],
                                     device=device, dtype=pseudo_labels.dtype)

    return new_pseudo_labels

