#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn
from clusten import CLUSTENWFFunction

"""
Test the correctness of WF custom kernel
"""

b = 256
n_ = 64
n = 196
m = 48
c = 32
ic = 4

# dummy data
weights = nn.Parameter(torch.randn(b, n_, m, ic)).cuda()
weights.retain_grad()
feat = nn.Parameter(torch.randn(b, n, c)).cuda()
feat.retain_grad()
nn_idx = torch.randint(n, (b, n_, m)).cuda()

# use the custom kernel
feat_new = CLUSTENWFFunction.apply(weights, feat, nn_idx)
feat_new.mean().backward()
grad_weights = weights.grad.clone().detach()
weights.grad.data.zero_()
grad_feat = feat.grad.clone().detach()
feat.grad.data.zero_()

# use the pytorch equivalent
'''
feat_new2 = CLUSTENWFFunction.apply(weights,feat,nn_idx)
'''
feat_gather = feat.gather(index=nn_idx.reshape(b, -1, 1).expand(-1, -1, c), dim=1).reshape(b, n_, m, c)
feat_new2 = weights.transpose(-1, -2) @ feat_gather
feat_new2.mean().backward()
grad_weights2 = weights.grad.clone().detach()
weights.grad.data.zero_()
grad_feat2 = feat.grad.clone().detach()
feat.grad.data.zero_()


print('diff of forward: ', torch.linalg.norm(feat_new2 - feat_new))
print('diff of grad weights: ', torch.linalg.norm(grad_weights2 - grad_weights))
print('diff of grad feat: ', torch.linalg.norm(grad_feat2 - grad_feat))
