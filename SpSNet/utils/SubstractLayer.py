# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet as scn
from torch.nn import Module
import torch
import numpy as np
import time
class DistMatchLayer_v4(Module):
    def __init__(self, dimension, nPlanes,full_scale = 32,topk = 5,r = 0.5):
        super(DistMatchLayer_v4, self).__init__()
        self.dimension = dimension
        self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.r = r

    def pairwise_distances_cos(self, x, y=None):
        return 1 - torch.clamp(
            torch.mm(x, torch.transpose(y, 0, 1))
            / torch.norm(x, dim=1).view(-1, 1)
            / torch.norm(y, dim=1).view(1, -1),
            0.0,
            np.inf,
        )
    def pairwise_distances_l2(self,x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.sqrt(torch.clamp(dist, 0.0, np.inf))/self.full_scale

    def forward(self, a, b):
        # assert input.features.shape[-1] == vecter.shape[-1]
        point_idx_a = a.get_spatial_locations()[:, -1]
        point_idx_b = b.get_spatial_locations()[:, -1]
        max_idx = torch.max(point_idx_a).max().item()
        coordinate_a = a.get_spatial_locations()[:, :-1]
        coordinate_b = b.get_spatial_locations()[:, :-1]
        output = scn.SparseConvNetTensor()
        output.metadata = a.metadata
        output.spatial_size = a.spatial_size
        tmp = []
        for i in range(max_idx + 1):
            if self.topk > coordinate_b[point_idx_b == i].size()[0]:
                topk = coordinate_b[point_idx_b == i].size()[0]
            else:
                topk = self.topk
            dist_c = self.pairwise_distances_l2(
                coordinate_a[point_idx_a == i].type(torch.cuda.FloatTensor),
                coordinate_b[point_idx_b == i].type(torch.cuda.FloatTensor),
            )
            dist = dist_c
            tmp_f = []
            # dist_w, idx_pick = torch.topk(dist, topk)#

            idx_pick = torch.argsort(dist, 1)[:,:topk]
            dist_w = torch.gather(dist_c,1,idx_pick)
            dist_w = (self.r - torch.clamp(dist_w, 0., self.r))

            tmp_b = b.features[point_idx_b == i][idx_pick.reshape(-1)].reshape(-1,topk,b.features.size()[-1]) *dist_w.view(-1,topk,1)
            tmp_b = torch.sum(tmp_b,1)
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        output.features = torch.cat([a.features, tmp.cuda()], 1)
        return output

    def __repr__(self):
        s = "DistMatchLayer"
        return s