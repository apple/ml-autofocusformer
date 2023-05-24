#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn
from clusten import CLUSTENAVFunction

"""
Test the correctness of AV custom kernel
"""

b = 256
h = 4
n = 196
m = 48
c = 32

# dummy data
attn = nn.Parameter(torch.randn(b, h, n, m)).cuda()
attn.retain_grad()
val = nn.Parameter(torch.randn(b, h, n, c)).cuda()
val.retain_grad()
nn_idx = torch.randint(n, (b, n, m)).cuda()

# use the custom kernel
feat = CLUSTENAVFunction.apply(attn, val, nn_idx)
feat.mean().backward()
grad_attn = attn.grad.clone().detach()
attn.grad.data.zero_()
grad_val = val.grad.clone().detach()
val.grad.data.zero_()

# use the pytorch equivalent
'''
feat2 = CLUSTENAVFunction.apply(attn,val,nn_idx)
'''
val_gather = val.gather(index=nn_idx.reshape(b, 1, -1, 1).expand(-1, h, -1, c), dim=2).reshape(b, h, n, m, c)
feat2 = (attn.unsqueeze(4) * val_gather).sum(3)
feat2.mean().backward()
grad_attn2 = attn.grad.clone().detach()
attn.grad.data.zero_()
grad_val2 = val.grad.clone().detach()
val.grad.data.zero_()


print('diff of forward: ', torch.linalg.norm(feat2 - feat))
print('diff of grad attn: ', torch.linalg.norm(grad_attn2 - grad_attn))
print('diff of grad val: ', torch.linalg.norm(grad_val2 - grad_val))
