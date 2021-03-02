# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet as scn
from torch.nn import Module
import torch


class GlobalJoinLayer(Module):
    def __init__(self, dimension):
        super(GlobalJoinLayer, self).__init__()
        self.dimension = dimension

    def forward(self, input, vecter):
        assert input.features.shape[-1] == vecter.shape[-1]
        point_idx = input.get_spatial_locations()[:, -1]
        output = scn.SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = torch.cat([input.features, vecter[point_idx]], -1)
        return output

    def __repr__(self):
        s = "GlobalAddLayer"
        return s


class GlobalAddLayer(Module):
    def __init__(self, dimension):
        super(GlobalAddLayer, self).__init__()
        self.dimension = dimension

    def forward(self, input, vecter):
        assert input.features.shape[-1] == vecter.shape[-1]
        point_idx = input.get_spatial_locations()[:, -1]
        output = scn.SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = input.features + vecter[point_idx]
        return output

    def __repr__(self):
        s = "GlobalAddLayer"
        return s


class GlobalMaskLayer(Module):
    def __init__(self, dimension):
        super(GlobalMaskLayer, self).__init__()
        self.dimension = dimension

    def forward(self, input, vecter):
        assert input.features.shape[-1] == vecter.shape[-1]
        point_idx = input.get_spatial_locations()[:, -1]
        output = scn.SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = input.features * vecter[point_idx]
        return output

    def __repr__(self):
        s = "GlobalAddLayer"
        return s

class GlobalMaskLayer_v2(Module):
    def __init__(self, dimension):
        super(GlobalMaskLayer_v2, self).__init__()
        self.dimension = dimension

    def forward(self, input, vecter):
        assert input.features.shape[-1] == vecter.shape[-1]
        point_idx = input.get_spatial_locations()[:, -1]
        output = scn.SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = input.features * torch.sigmoid(vecter[point_idx])*2
        return output

    def __repr__(self):
        s = "GlobalAddLayer"
        return s

class GlobalMaskLayer_v3(Module):
    def __init__(self, dimension):
        super(GlobalMaskLayer_v3, self).__init__()
        self.dimension = dimension

    def forward(self, input, vecter):
        assert input.features.shape[-1] == vecter.shape[-1]
        point_idx = input.get_spatial_locations()[:, -1]
        output = scn.SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        vecter = torch.softmax(vecter,1)
        output.features = input.features * vecter[point_idx]#torch.softmax(vecter[point_idx],1)
        return output

    def __repr__(self):
        s = "GlobalAddLayer"
        return s

class GlobalMaskLayer_v4(Module):
    def __init__(self, dimension):
        super(GlobalMaskLayer_v4, self).__init__()
        self.dimension = dimension

    def forward(self, input, vecter):
        assert input.features.shape[-1] == vecter.shape[-1]
        point_idx = input.get_spatial_locations()[:, -1]
        output = scn.SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        # vecter = torch.softmax(vecter,1)
        output.features = input.features * torch.sigmoid(vecter[point_idx])*2 #torch.softmax(vecter[point_idx],1)
        # output.features = input.features * torch.softmax(vecter[point_idx],1) #torch.softmax(vecter[point_idx],1)
        return output

    def __repr__(self):
        s = "GlobalAddLayer"
        return s

class GlobalMaskLayer_v5(Module):
    def __init__(self, dimension):
        super(GlobalMaskLayer_v5, self).__init__()
        self.dimension = dimension

    def forward(self, input, vecter):
        assert input.features.shape[-1] == vecter.shape[-1]
        point_idx = input.get_spatial_locations()[:, -1]
        output = scn.SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        vecter = torch.softmax(vecter,1) + 1
        output.features = input.features * vecter[point_idx]#torch.softmax(vecter[point_idx],1)
        return output

    def __repr__(self):
        s = "GlobalAddLayer"
        return s

class GlobalMaskLayer_combine(Module):
    def __init__(self, dimension):
        super(GlobalMaskLayer_combine, self).__init__()
        self.dimension = dimension

    def forward(self, input, vecter):
        assert input.features.shape[-1] == vecter.shape[-1]
        point_idx = input.get_spatial_locations()[:, -1]
        output = scn.SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = torch.cat(input.features, vecter[point_idx], 1)
        return output

    def __repr__(self):
        s = "GlobalAddLayer"
        return s

class GlobalSplitLayer(Module):
    def __init__(self, dimension, full_scale, frames=2):
        super(GlobalSplitLayer, self).__init__()
        self.dimension = dimension
        self.frames = frames
        self.input1 = scn.InputLayer(dimension, full_scale, mode=4)
        self.input2 = scn.InputLayer(dimension, full_scale, mode=4)

    def forward(self, input):
        point_idx_in = input.get_spatial_locations()[:, -1]
        max_idx = torch.max(point_idx_in).max().item()
        feature = input.features
        cood = input.get_spatial_locations()
        assert max_idx % self.frames
        batch_size = (max_idx + 1) // 2
        tmp = cood[point_idx_in >= batch_size]
        tmp[:, -1] = tmp[:, -1] - batch_size

        return (
            self.input1(
                [cood[point_idx_in < batch_size], feature[point_idx_in < batch_size]]
            ),
            self.input2([tmp, feature[point_idx_in >= batch_size]]),
        )

    def __repr__(self):
        s = "GlobalSplitLayer"
        return s


class GlobalMergeLayer(Module):
    def __init__(self, dimension, full_scale, frames=2):
        super(GlobalMergeLayer, self).__init__()
        self.dimension = dimension
        self.frames = frames
        self.input = scn.InputLayer(dimension, full_scale, mode=4)

    def forward(self, input1, input2):
        point_idx_in = input1.get_spatial_locations()[:, -1]
        max_idx = torch.max(point_idx_in).max().item()
        assert max_idx % self.frames
        batch_size = 1 + max_idx
        tmp_f = torch.concat([input1.features, input2.features])
        tmp_c = input2.get_spatial_locations()
        tmp_c[:, -1] = tmp_c[:, -1] + batch_size
        tmp_c = torch.concat([input1.get_spatial_locations(), tmp_c])
        return self.input([tmp_c, tmp_f])

    def __repr__(self):
        s = "GlobalMergeLayer"
        return s
