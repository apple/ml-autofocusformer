#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import math
import torch


def points2img(pos, pixel, h, w):
    """
    Scatter tokens onto a canvas of size h x w
    Args:
        pos - b x n x 2, position of tokens, should be valid indices in the canvas
        pixel - b x n x c, feature of tokens
        h,w - int, height and width of the canvas
    Returns:
        img - b x c x h x w, the resulting grid img; blank spots filled with 0
    """
    b, n, c = pixel.shape
    img = torch.zeros(b, h*w, c, device=pos.device).to(pixel.dtype)
    idx = (pos[:, :, 1]*w+pos[:, :, 0]).long().unsqueeze(2).expand(-1, -1, c)  # b x n x c
    img = img.scatter(src=pixel, index=idx, dim=1)
    return img.permute(0, 2, 1).reshape(b, c, h, w)


def knn_keops(query, database, k, return_dist=False):
    """
    Compute k-nearest neighbors using the Keops library
    Backward pass turned off; Keops does not provide backward pass for distance
    Args:
        query - b x n_ x c, the position of tokens looking for knn
        database - b x n x c, the candidate tokens for knn
        k - int, the nunmber of neighbors to be found
        return_dist - bool, whether to return distance to the neighbors
    Returns:
        nn_dix - b x n x k, the indices of the knn
        nn_dist - b x n x k, if return_dist, the distance to the knn
    """
    b, n, c = database.shape
    with torch.no_grad():
        query = query.detach()
        database = database.detach()
        # Keops does not support half precision
        if query.dtype != torch.float32:
            query = query.to(torch.float32)
        if database.dtype != torch.float32:
            database = database.to(torch.float32)
        from pykeops.torch import LazyTensor
        query_ = LazyTensor(query[:, None, :, :])
        database_ = LazyTensor(database[:, :, None, :])
        dist = ((query_-database_) ** 2).sum(-1) ** 0.5  # b x n x n_
    if return_dist:
        nn_dist, nn_idx = dist.Kmin_argKmin(k, dim=1)  # b x n_ x k
        return nn_idx, nn_dist
    else:
        nn_idx = dist.argKmin(k, dim=1)  # b x n_ x k
        return nn_idx


def space_filling_cluster(pos, m, h, w, no_reorder=False, sf_type='', use_anchor=True):
    """
    The balanced clustering algorithm based on space-filling curves
    In the case where number of tokens not divisible by cluster size,
    the last cluster will have a few blank spots, indicated by the mask returned
    Args:
        pos - b x n x 2, positions of tokens
        m - int, target size of the clusters
        h,w - int, height and width
        no_reorder - bool, if True, return the clustering based on the original order of tokens;
                            otherwise, reorder the tokens so that the same cluster stays together
        sf_type - str, can be 'peano' or 'hilbert', or otherwise, horizontal scanlines w/ alternating
                        direction in each row by default
        use_anchor - bool, whether to use space-fiiling anchors or not; if False, directly compute
                            space-filling curves on the token positions
    Returns:
        pos - b x n x 2, returned only if no_reorder is False; the reordered position of tokens
        cluster_mean_pos - b x k x 2, the clustering centers
        member_idx - b x k x m, the indices of tokens in each cluster
        cluster_mask - b x k x m, the binary mask indicating the paddings in last cluster (0 if padding)
        pos_ranking - b x n x 1, returned only if no_reorder is False; i-th entry is the idx of the token
                                rank i in the new order
    """
    with torch.no_grad():
        pos = pos.detach()

        if pos.dtype != torch.float:
            pos = pos.to(torch.float)
        b, n, d = pos.shape

        k = int(math.ceil(n/m))

        if use_anchor:
            patch_len = (h*w/k)**0.5
            num_patch_h = int(round(h / patch_len))
            num_patch_w = int(round(w / patch_len))
            patch_len_h, patch_len_w = h / num_patch_h, w / num_patch_w
            if sf_type == 'peano':
                num_patch_h = max(3, int(3**round(math.log(num_patch_h, 3))))
                patch_len_h = h / num_patch_h
                num_patch_w = int(round(w / h * 3) * (num_patch_h / 3))
                patch_len_w = w / num_patch_w
            elif sf_type == 'hilbert':
                num_patch_h = max(2, int(2**round(math.log(num_patch_h, 2))))
                patch_len_h = h / num_patch_h
                num_patch_w = int(round(w / h * 2) * (num_patch_h / 2))
                patch_len_w = w / num_patch_w
            hs = torch.arange(0, num_patch_h, device=pos.device)
            ws = torch.arange(0, num_patch_w, device=pos.device)
            ys, xs = torch.meshgrid(hs, ws)
            grid_pos = torch.stack([xs, ys], dim=2)  # h x w x 2
            grid_pos = grid_pos.reshape(-1, 2)

            # sort the grid centers to one line
            if sf_type == 'peano':
                order_grid_idx, order_idx = calculate_peano_order(num_patch_h, num_patch_w, grid_pos.unsqueeze(0))
                order_grid_idx = order_grid_idx[0]
                order_idx = order_idx[0]
            elif sf_type == 'hilbert':
                order_grid_idx, order_idx = calculate_hilbert_order(num_patch_h, num_patch_w, grid_pos.unsqueeze(0))
                order_grid_idx = order_grid_idx[0]
                order_idx = order_idx[0]
            else:
                order_mask = torch.ones_like(ys)  # h x w
                order_mask[1::2] = -1
                order_mask = order_mask * xs
                order_mask = order_mask + ys*w
                order_mask[1::2] += (w-1)
                order_mask = order_mask.reshape(-1)
                order_idx = order_mask.sort()[1]
                order_idx_src = torch.arange(len(order_idx)).to(pos.device)
                order_grid_idx = torch.zeros_like(order_idx_src)
                order_grid_idx.scatter_(index=order_idx, dim=0, src=order_idx_src)

            ordered_grid = grid_pos[order_idx]
            patch_len_hw = torch.Tensor([patch_len_w, patch_len_h]).to(pos.device)

            init_pos_means = ordered_grid * patch_len_hw + patch_len_hw/2 - 0.5
            nump = ordered_grid.shape[0]

            prev_means = torch.zeros_like(init_pos_means)
            prev_means[1:] = init_pos_means[:nump-1].clone()
            prev_means[0] = prev_means[1] - (prev_means[2]-prev_means[1])  # float('inf')
            next_means = torch.zeros_like(init_pos_means)
            next_means[:nump-1] = init_pos_means[1:].clone()
            next_means[-1] = next_means[-2] + (next_means[-2]-next_means[-3])  # float('inf')

            mean_assignment = (pos / patch_len_hw).floor()
            mean_assignment = mean_assignment[..., 0] + mean_assignment[..., 1] * num_patch_w
            mean_assignment = order_grid_idx.unsqueeze(0).expand(b, -1).gather(index=mean_assignment.long(), dim=1).unsqueeze(2)  # b x n x 1

            prev_mean_assign = prev_means.unsqueeze(0).expand(b, -1, -1).gather(index=mean_assignment.expand(-1, -1, d), dim=1)  # b x n x d
            next_mean_assign = next_means.unsqueeze(0).expand(b, -1, -1).gather(index=mean_assignment.expand(-1, -1, d), dim=1)  # b x n x d
            dist_prev = (pos-prev_mean_assign).pow(2).sum(-1)  # b x n
            dist_next = (pos-next_mean_assign).pow(2).sum(-1)
            dist_ratio = dist_prev / (dist_next + 1e-5)

            pos_ranking = mean_assignment * (dist_ratio.max()+1) + dist_ratio.unsqueeze(2)
            pos_ranking = pos_ranking.sort(dim=1)[1]  # b x n x 1

        else:
            if sf_type == 'peano':
                _, pos_ranking = calculate_peano_order(h, w, pos)
            elif sf_type == 'hilbert':
                _, pos_ranking = calculate_hilbert_order(h, w, pos)
            else:
                hs = torch.arange(0, h, device=pos.device)
                ws = torch.arange(0, w, device=pos.device)
                ys, xs = torch.meshgrid(hs, ws)
                order_mask = torch.ones_like(ys)  # h x w
                order_mask[1::2] = -1
                order_mask = order_mask * xs
                order_mask = order_mask + ys*w
                order_mask[1::2] += (w-1)
                order_mask = order_mask.reshape(-1)
                pos_idx = pos[..., 0] + pos[..., 1] * w
                order_mask = order_mask.gather(index=pos_idx.long().reshape(-1), dim=0).reshape(b, n)
                pos_ranking = order_mask.sort()[1]
            pos_ranking = pos_ranking.unsqueeze(2)

        pos = pos.gather(index=pos_ranking.expand(-1, -1, d), dim=1)  # b x n x d

        if k*m == n:
            cluster_mask = None
            cluster_mean_pos = pos.reshape(b, k, -1, d).mean(2)
        else:
            pos_pad = torch.zeros(b, k*m, d, dtype=pos.dtype, device=pos.device)
            pos_pad[:, :n] = pos.clone()
            cluster_mask = torch.zeros(b, k*m, device=pos.device).long()
            cluster_mask[:, :n] = 1
            cluster_mask = cluster_mask.reshape(b, k, m)
            cluster_mean_pos = pos_pad.reshape(b, k, -1, d).sum(2) / cluster_mask.sum(2, keepdim=True)

        if no_reorder:
            if k*m == n:
                member_idx = pos_ranking.reshape(b, k, m)
            else:
                member_idx = torch.zeros(b, k*m, device=pos.device, dtype=torch.int64)
                member_idx[:, :n] = pos_ranking.squeeze(2)
                member_idx = member_idx.reshape(b, k, m)
            return cluster_mean_pos, member_idx, cluster_mask
        else:
            member_idx = torch.arange(k*m, device=pos.device)
            member_idx[n:] = 0
            member_idx = member_idx.unsqueeze(0).expand(b, -1)  # b x k*m
            member_idx = member_idx.reshape(b, k, m)

            return pos, cluster_mean_pos, member_idx, cluster_mask, pos_ranking


def calculate_peano_order(h, w, pos):
    """
    Given height and width of the canvas and position of tokens,
    calculate the peano curve order of the tokens
    Args:
        h,w - int, height and width
        pos - b x n x 2, positions of tokens
    Returns:
        final_order_ - b x n, i-th entry is the rank of i-th token in the new order
        final_order_index - b x n, i-th entry is the idx of the token rank i in the new order
    """
    b, n, _ = pos.shape
    num_levels = math.ceil(math.log(h, 3))
    assert num_levels >= 1, "h too short"
    first_w = None
    if h != w:
        first_w = round(3 * (w/h))
        if first_w == 3:
            first_w = None
    init_dict = torch.Tensor([[2, 3, 8], [1, 4, 7], [0, 5, 6]]).to(pos.device)
    inverse_dict = torch.Tensor([[[1, 1], [1, -1], [1, 1]], [[-1, 1], [-1, -1], [-1, 1]], [[1, 1], [1, -1], [1, 1]]]).to(pos.device)
    if first_w is not None:
        init_dict_flip = init_dict.flip(dims=[0])
        init_dict_f = torch.cat([init_dict, init_dict_flip], dim=1)  # 3 x 6
        init_dict_f = init_dict_f.repeat(1, math.ceil(first_w/6))
        init_dict_f = init_dict_f[:, :first_w]  # 3 x fw
        w_index = torch.arange(math.ceil(first_w/3)).to(pos.device).repeat_interleave(3)[:first_w] * 9  # fw
        init_dict_f = init_dict_f + w_index
        init_dict_f = init_dict_f.reshape(-1)  # 3*fw
        inverse_dict_f = inverse_dict[:, :2].repeat(1, math.ceil(first_w/2), 1)[:, :first_w]  # 3 x fw x 2
        inverse_dict_f = inverse_dict_f.reshape(-1, 2)
    init_dict = init_dict.reshape(-1)  # 9
    inverse_dict = inverse_dict.reshape(-1, 2)  # 9 x 2
    last_h = h
    rem_pos = pos
    levels_pos = []
    for le in range(num_levels):
        cur_h = last_h / 3
        level_pos = (rem_pos / cur_h).floor()
        levels_pos.append(level_pos)
        rem_pos = rem_pos % cur_h
        last_h = cur_h
    orders = []
    for i in range(len(levels_pos)):
        inverse = torch.ones_like(pos)  # b x n x 2
        for j in range(i):
            cur_level_pos = levels_pos[i-j-1]
            if i-j-1 == 0 and first_w is not None:
                cur_level_pos_index = cur_level_pos[..., 0] + cur_level_pos[..., 1] * first_w  # b x n
                cur_inverse = inverse_dict_f.gather(index=cur_level_pos_index.long().view(-1, 1).expand(-1, 2), dim=0).reshape(b, n, 2)
            else:
                cur_level_pos_index = cur_level_pos[..., 0] + cur_level_pos[..., 1] * 3  # b x n
                cur_inverse = inverse_dict.gather(index=cur_level_pos_index.long().view(-1, 1).expand(-1, 2), dim=0).reshape(b, n, 2)
            inverse = cur_inverse * inverse
        level_pos = levels_pos[i]
        inversed_pos = torch.where(inverse > 0, level_pos, 2-level_pos)
        if i == 0 and first_w is not None:
            inversed_pos_index = inversed_pos[..., 0] + inversed_pos[..., 1] * first_w  # b x n
            cur_order = init_dict_f.gather(index=inversed_pos_index.long().view(-1), dim=0).reshape(b, n)
        else:
            inversed_pos_index = inversed_pos[..., 0] + inversed_pos[..., 1] * 3  # b x n
            cur_order = init_dict.gather(index=inversed_pos_index.long().view(-1), dim=0).reshape(b, n)
        orders.append(cur_order)
    final_order = orders[-1]
    for i in range(len(orders)-1):
        cur_order = orders[i]
        final_order = final_order + cur_order * (9**(num_levels-i-1))
    final_order_index = final_order.sort(dim=1)[1]
    order_src = torch.arange(n).to(pos.device).unsqueeze(0).expand(b, -1)  # b x n
    final_order_ = torch.zeros_like(order_src)
    final_order_.scatter_(index=final_order_index, src=order_src, dim=1)
    return final_order_, final_order_index


def calculate_hilbert_order(h, w, pos):
    """
    Given height and width of the canvas and position of tokens,
    calculate the hilber curve order of the tokens
    Args:
        h,w - int, height and width
        pos - b x n x 2, positions of tokens
    Returns:
        final_order_ - b x n, i-th entry is the rank of i-th token in the new order
        final_order_index - b x n, i-th entry is the idx of the token rank i in the new order
    """
    b, n, _ = pos.shape
    num_levels = math.ceil(math.log(h, 2))
    assert num_levels >= 1, "h too short"
    first_w = None
    if h != w:
        first_w = round(2 * (w/h))
        if first_w == 2:
            first_w = None
    rotate_dict = torch.Tensor([[[-1, 1], [0, 0]], [[0, -1], [0, 1]], [[1, 0], [-1, 0]]]).to(pos.device)  # 3 x 2 x 2 -1 means left, 1 means right
    if first_w is not None:
        rotate_dict_f = rotate_dict[0].repeat(1, math.ceil(first_w/2))[:, :first_w]  # 2 x fw
        rotate_dict_f = rotate_dict_f.reshape(-1)  # 2*fw
    rotate_dict = rotate_dict.reshape(3, -1)  # 3 x 4
    rot_res_dict = torch.Tensor([[0, 3, 1, 2], [2, 3, 1, 0], [2, 1, 3, 0], [0, 1, 3, 2]]).to(pos.device)  # 4 x 4
    last_h = h
    rem_pos = pos
    levels_pos = []
    for le in range(num_levels):
        cur_h = last_h / 2
        level_pos = (rem_pos / cur_h).floor()
        levels_pos.append(level_pos)
        rem_pos = rem_pos % cur_h
        last_h = cur_h
    orders = []
    for i in range(len(levels_pos)):
        level_pos = levels_pos[i]
        if i == 0 and first_w is not None:
            level_pos_index = level_pos[..., 0] + level_pos[..., 1] * first_w  # b x n
        else:
            level_pos_index = level_pos[..., 0] + level_pos[..., 1] * 2  # b x n
        rotate = torch.zeros_like(pos[..., 0])
        for j in range(i):
            cur_level_pos = levels_pos[j]
            if j == 0 and first_w is not None:
                cur_level_pos_index = cur_level_pos[..., 0] + cur_level_pos[..., 1] * first_w  # b x n
                cur_rotate = rotate_dict_f.gather(index=cur_level_pos_index.long().view(-1), dim=0).reshape(b, n)
            else:
                rotate_d = rotate_dict.gather(index=(rotate % 3).long().view(-1, 1).expand(-1, 4), dim=0).reshape(b, n, 4)
                cur_level_pos_index = cur_level_pos[..., 0] + cur_level_pos[..., 1] * 2  # b x n
                cur_rotate = rotate_d.gather(index=cur_level_pos_index.long().unsqueeze(2), dim=2).reshape(b, n)
            rotate = cur_rotate + rotate
        rotate = rotate % 4
        rotate_res = rot_res_dict.gather(index=rotate.long().view(-1, 1).expand(-1, 4), dim=0).reshape(b, n, 4)
        rotate_res = rotate_res.gather(index=level_pos_index.long().unsqueeze(2), dim=2).squeeze(2)  # b x n
        orders.append(rotate_res)
    final_order = orders[-1]
    for i in range(len(orders)-1):
        cur_order = orders[i]
        final_order = final_order + cur_order * (4**(num_levels-i-1))
    final_order_index = final_order.sort(dim=1)[1]
    order_src = torch.arange(n).to(pos.device).unsqueeze(0).expand(b, -1)  # b x n
    final_order_ = torch.zeros_like(order_src)
    final_order_.scatter_(index=final_order_index, src=order_src, dim=1)
    return final_order_, final_order_index
