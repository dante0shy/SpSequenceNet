# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet as scn
from torch.nn import Module
import torch

class GlobalMeanAttentionPooling(Module):
    def __init__(self, dimension):
        super(GlobalMeanAttentionPooling, self).__init__()
        self.dimension = dimension

    def forward(self, input):
        point_idx = input.get_spatial_locations()[:, -1]
        max_idx = torch.max(point_idx).max().item()
        feature = input.features
        output = []
        for i in range(max_idx + 1):
            output.append(torch.sigmoid(torch.mean(feature[point_idx == i], 0)))
        return torch.stack(output)

    def __repr__(self):
        s = "GlobalMaxPooling"
        return s

class GlobalMeanPooling(Module):
    def __init__(self, dimension):
        super(GlobalMeanPooling, self).__init__()
        self.dimension = dimension

    def forward(self, input):
        point_idx = input.get_spatial_locations()[:, -1]
        max_idx = torch.max(point_idx).max().item()
        feature = input.features
        output = []
        for i in range(max_idx + 1):
            output.append(torch.mean(feature[point_idx == i], 0))
        return torch.stack(output)

    def __repr__(self):
        s = "GlobalMeanPooling"
        return s
