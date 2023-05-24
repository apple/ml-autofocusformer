#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch
import numpy as np
from point_utils import space_filling_cluster
import cv2

"""
Test the correctness of the space_filling_cluster function
"""


def display_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


h, w = 100, 100  # canvas size
n = 2499  # number of tokens
m = 20  # cluster size
show_center = True

hs = torch.arange(0, h)
ws = torch.arange(0, w)
ys, xs = torch.meshgrid(hs, ws)
pos = torch.stack([xs, ys], dim=2).reshape(1, -1, 2)  # 1 x hw x 2

# random point cloud
pos = pos[:, torch.randperm(h*w)[:n]]  # 1 x n x 2

# cluster_mean_pos, member_idx, cluster_mask = space_filling_cluster(pos, m, h, w, no_reorder=True)
pos, cluster_mean_pos, member_idx, cluster_mask, _ = space_filling_cluster(pos, m, h, w, no_reorder=False)
if show_center:
    cluster_mean_pos = cluster_mean_pos.round().long()
k = member_idx.shape[1]  # number of clusters
print("n,k,m", n, k, m)
if cluster_mask is not None:
    cluster_mask = cluster_mask.reshape(1, -1, 1)
    print("cluster_mask invalid indices", (cluster_mask[0, :, 0] == 0).nonzero())

cluster_idx = torch.arange(k).view(1, -1, 1).expand(-1, -1, m).reshape(1, -1, 1)  # 1 x km x 1
mean_assignment = torch.zeros(1, n, 1, dtype=cluster_idx.dtype)
mean_assignment.scatter_(index=member_idx.reshape(1, -1, 1)[:, :n], dim=1, src=cluster_idx)

colors = torch.Tensor(np.random.uniform(size=(k, 3)))
ca = colors.gather(index=mean_assignment.reshape(-1, 1).expand(-1, 3), dim=0)  # n x 3
c = ca.shape[-1]

img = torch.zeros(h*w, c)
pos = pos[0]
idx = (pos[:, 1]*w+pos[:, 0]).long()  # n
idx = idx.unsqueeze(1).expand(-1, c)
img.scatter_(src=ca, index=idx, dim=0)
if show_center:
    cluster_mean_pos = cluster_mean_pos[0]
    center_idx = cluster_mean_pos[:, 1]*w+cluster_mean_pos[:, 0]
    img[center_idx] = torch.Tensor([0, 0, 1])  # cluster centers shown as red dots

img = img.reshape(h, w, c).numpy()
img = img.repeat(4, axis=0).repeat(4, axis=1)
# img = 1.0-img
display_img(img)
