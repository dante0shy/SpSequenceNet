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

class SubstractLayer(Module):
    def __init__(self, dimension, nPlanes):
        super(SubstractLayer, self).__init__()
        self.dimension = dimension
        self.s2d = scn.SparseToDense(dimension, nPlanes)
        # self.full_scale = full_scale

    def pairwise_distances_cos(self, x, y):
        """
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """

        return 1 - torch.clamp(
            torch.mm(x, torch.transpose(y, 0, 1))
            / torch.norm(x, dim=1).view(-1, 1)
            / torch.norm(y, dim=1).view(1, -1),
            0.0,
            np.inf,
        )
        # return torch.clamp(dist, 0.0, np.inf)

    # def pairwise_distances_l2(self,x, y):
    #     x_norm = (x ** 2).sum(1).view(-1, 1)
    #     y_t = torch.transpose(y, 0, 1)
    #     y_norm = (y ** 2).sum(1).view(1, -1)
    #     dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    #     return torch.sqrt(torch.clamp(dist, 0.0, np.inf))/self.full_scale

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
            dist_c = self.pairwise_distances_cos(
                coordinate_a[point_idx_a == i].type(torch.cuda.FloatTensor),
                coordinate_b[point_idx_b == i].type(torch.cuda.FloatTensor),
            )
            dist_f = self.pairwise_distances_cos(
                a.features[point_idx_a == i], b.features[point_idx_b == i]
            )
            dist = dist_c * dist_f
            idx_pick = torch.argmin(dist, 1)
            tmp.append(b.features[point_idx_b == i][idx_pick])
            # output.features =  a.features- tmp[point_idx,coordinate[:,0],coordinate[:,1]]
            pass
        tmp = torch.cat(tmp, 0)
        output.features = torch.cat([a.features, tmp.cuda()], 1)
        return output

    def __repr__(self):
        s = "GlobalAddLayer"
        return s


class DistMatchLayer_v2(Module):
    def __init__(self, dimension, nPlanes,full_scale = 32):
        super(DistMatchLayer_v2, self).__init__()
        self.dimension = dimension
        self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale

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
            dist_c = self.pairwise_distances_l2(
                coordinate_a[point_idx_a == i].type(torch.cuda.FloatTensor),
                coordinate_b[point_idx_b == i].type(torch.cuda.FloatTensor),
            )
            dist_f = self.pairwise_distances_cos(
                a.features[point_idx_a == i], b.features[point_idx_b == i]
            )
            dist = dist_c * dist_f
            idx_pick = torch.argmin(dist, 1)

            dist_w = torch.clamp(  torch.gather(dist_c, 1, idx_pick.view(-1, 1)), 0., 0.5).view(-1, 1)
            tmp.append(b.features[point_idx_b == i][idx_pick]*(0.5 - dist_w)*2)
            # output.features =  a.features- tmp[point_idx,coordinate[:,0],coordinate[:,1]]
            pass
        tmp = torch.cat(tmp, 0)
        output.features = torch.cat([a.features, tmp.cuda()], 1)
        return output

    def __repr__(self):
        s = "DistMatchLayer"
        return s
class DistMatchLayer_v3_2(Module):
    def __init__(self, dimension, nPlanes, full_scale=32):
        super(DistMatchLayer_v3_2, self).__init__()
        self.dimension = dimension
        self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.fc_1 = torch.nn.Linear(nPlanes,1)

    def pairwise_distances_cos(self, x, y=None):
        return 1 - torch.clamp(
            torch.mm(x, torch.transpose(y, 0, 1))
            / torch.norm(x, dim=1).view(-1, 1)
            / torch.norm(y, dim=1).view(1, -1),
            0.0,
            np.inf,
        )

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / self.full_scale

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
            dist_c = self.pairwise_distances_l2(
                coordinate_a[point_idx_a == i].type(torch.cuda.FloatTensor),
                coordinate_b[point_idx_b == i].type(torch.cuda.FloatTensor),
            )
            dist = dist_c
            dist = torch.clamp(dist, 0., 0.5)#.view(-1, 1)
            tmp_f = []
            idx_pick = torch.argsort(dist, 1)

            tmp_d = (0.5- torch.gather(dist, 1, idx_pick)[:,:5]) * 2


            tmp_f = b.features[point_idx_b == i][idx_pick[:,:5].flatten()] #.view(-1,5,112)

            tmp_weights = torch.sigmoid(self.fc_1(tmp_f)).view(-1,5)
            # a.repeat(3, 1, 1, 1)
            try:
                tmp_d = tmp_d * tmp_weights
            except:
                pass
            # print(tmp_d.shape)
            # print(tmp_f.shape)
            tmp_f = tmp_f.view(-1,5,112)
            tmp_d = tmp_d.view(-1,5,1)
            # print(tmp_d.repeat(tmp_d.shape[0], 5, 112).shape)
            tmp.append(torch.sum(tmp_f * tmp_d,1))
            # for idx in range(dist.size()[0]):
            #     idx_pick = torch.argsort(dist[idx, :])[:5]
            #     # idx_pick = idx_pick[dist[i][idx_pick] <= 0.5 ]
            #     # num =dist[i][idx_pick] <= 0.5
            #     dist_sub = torch.clamp(dist[idx][idx_pick], 0., 0.5).view(-1, 1)
            #     # if idx_pick.size()[0]:
            #
            #     tmp_f.append(torch.sum(b.features[point_idx_b == i][idx_pick, :] * (0.5 - dist_sub) * 2, 0).view(1, -1))
            #
            # tmp.extend(tmp_f)
            # output.features =  a.features- tmp[point_idx,coordinate[:,0],coordinate[:,1]]
            pass
        tmp = torch.cat(tmp, 0)
        output.features = torch.cat([a.features, tmp.cuda()], 1)
        return output

    def __repr__(self):
        s = "DistMatchLayer"
        return s

class DistMatchLayer_v3(Module):
    def __init__(self, dimension, nPlanes,full_scale = 32):
        super(DistMatchLayer_v3, self).__init__()
        self.dimension = dimension
        self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale

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
            dist_c = self.pairwise_distances_l2(
                coordinate_a[point_idx_a == i].type(torch.cuda.FloatTensor),
                coordinate_b[point_idx_b == i].type(torch.cuda.FloatTensor),
            )
            dist = dist_c
            tmp_f = []
            idx_pick = torch.argmin(dist, 1)
            tmp.append(b.features[point_idx_b == i][idx_pick])
            for idx in range(dist.size()[0]):
                idx_pick = torch.argsort(dist[idx,:])[:5]
                # idx_pick = idx_pick[dist[i][idx_pick] <= 0.5 ]
                # num =dist[i][idx_pick] <= 0.5
                dist_sub =  torch.clamp(dist[idx][idx_pick],0.,0.5) .view(-1,1)
                # if idx_pick.size()[0]:

                tmp_f.append(torch.sum(b.features[point_idx_b == i][idx_pick,:]*(0.5-dist_sub)*2,0).view(1,-1))

            tmp.extend(tmp_f)
            # output.features =  a.features- tmp[point_idx,coordinate[:,0],coordinate[:,1]]
            pass
        tmp = torch.cat(tmp, 0)
        output.features = torch.cat([a.features, tmp.cuda()], 1)
        return output

    def __repr__(self):
        s = "DistMatchLayer"
        return s

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

class DistMatchLayer_v4_mutli(Module):
    def __init__(self, dimension, nPlanes,full_scale = 32,topk = 5,r = 0.5):
        super(DistMatchLayer_v4_mutli, self).__init__()
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
        s = "DistMatchLayer_v4_mutli"
        return s


class DistMatchLayer_v4_5(Module):
    def __init__(self, dimension, nPlanes,full_scale = 32,topk = 3,r = 0.5):
        super(DistMatchLayer_v4_5, self).__init__()
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
            dist_w = (self.r - torch.clamp(dist_w, 0., self.r)) * (1/self.r)

            tmp_b = b.features[point_idx_b == i][idx_pick.reshape(-1)].reshape(-1,topk,b.features.size()[-1]) *dist_w.view(-1,topk,1)
            tmp_b = torch.sum(tmp_b,1)
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        output.features = torch.cat([a.features, tmp.cuda()], 1)
        return output

    def __repr__(self):
        s = "DistMatchLayer"
        return s

class DistMatchLayer_v5(Module):
    def __init__(self, dimension, nPlanes,full_scale = 32,topk = 5):
        super(DistMatchLayer_v5, self).__init__()
        self.dimension = dimension
        self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk

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
            # dist_c = self.pairwise_distances_l2(
            #     coordinate_a[point_idx_a == i].type(torch.cuda.FloatTensor),
            #     coordinate_b[point_idx_b == i].type(torch.cuda.FloatTensor),
            # )

            # dist_f = torch.matmul(a.features[point_idx_a == i],b.features[point_idx_b == i].T)
            dist_f = torch.softmax(torch.matmul(a.features[point_idx_a == i],b.features[point_idx_b == i].T),1)
            dist_c = self.pairwise_distances_l2(
                    coordinate_a[point_idx_a == i].type(torch.cuda.FloatTensor),
                    coordinate_b[point_idx_b == i].type(torch.cuda.FloatTensor),
            )
            dist_c = (1.0 - torch.clamp(dist_c, 0., 1.0))

            dist = dist_f * dist_c
            # tmp_f = []
            # dist_w, idx_pick = torch.topk(dist, topk)#

            # idx_pick = torch.argsort(dist, 1)[:,:topk]
            # dist_w = torch.gather(dist_c,1,idx_pick)
            # dist_w = (1.0 - torch.clamp(dist_w, 0., 1.0))
            tmp_b =torch.matmul(dist,b.features[point_idx_b == i])
            # tmp_b = b.features[point_idx_b == i][idx_pick.reshape(-1)].reshape(-1,topk,b.features.size()[-1]) *dist_w.view(-1,topk,1)
            # tmp_b = torch.max(tmp_b,1)[0]
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        output.features = torch.cat([a.features, tmp.cuda()], 1)
        return output

    def __repr__(self):
        s = "DistMatchLayer"
        return s

class DistMatchLayer_v6(Module):
    def __init__(self, dimension, nPlanes,full_scale = 32,topk = 5,r = 0.5):
        super(DistMatchLayer_v6, self).__init__()
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

class DistMatchLayer_v7(Module):
    def __init__(self, dimension, nPlanes,full_scale = 32,topk = 5):
        super(DistMatchLayer_v7, self).__init__()
        self.dimension = dimension
        self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.new_weight = torch.nn.Linear(2,1)
        # self.sigmoid = torch.sigmoid()

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
            # dist_c = self.pairwise_distances_l2(
            #     coordinate_a[point_idx_a == i].type(torch.cuda.FloatTensor),
            #     coordinate_b[point_idx_b == i].type(torch.cuda.FloatTensor),
            # )

            # dist_f = torch.matmul(a.features[point_idx_a == i],b.features[point_idx_b == i].T)
            dist_f = torch.softmax(torch.matmul(a.features[point_idx_a == i],b.features[point_idx_b == i].T),1)
            dist_c = self.pairwise_distances_l2(
                    coordinate_a[point_idx_a == i].type(torch.cuda.FloatTensor),
                    coordinate_b[point_idx_b == i].type(torch.cuda.FloatTensor),
            )
            # dist_c = (1.0 - torch.clamp(dist_c, 0., 1.0))
            len_f_a = dist_f.size(0)
            len_f_b = dist_f.size(1)
            # len_c = dist_c.size(0) * dist_c.size(1)
            dist = torch.sigmoid(self.new_weight(torch.cat([dist_f.view(-1,1), dist_c.view(-1,1)], 1))).reshape(len_f_a,len_f_b)
            # dist = dist_f * dist_c
            # tmp_f = []
            # dist_w, idx_pick = torch.topk(dist, topk)#

            idx_pick = torch.argsort(dist, 1)[:,:topk]
            dist_w = torch.gather(dist,1,idx_pick)
            # dist_w = (1.0 - torch.clamp(dist_w, 0., 1.0))
            # tmp_b =torch.matmul(dist,b.features[point_idx_b == i])
            tmp_b = b.features[point_idx_b == i][idx_pick.reshape(-1)].reshape(-1,topk,b.features.size()[-1]) *dist_w.view(-1,topk,1)
            tmp_b = torch.sum(tmp_b,1)
            # tmp_b = torch.max(tmp_b,1)[0]
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        output.features = torch.cat([a.features, tmp.cuda()], 1)
        return output

    def __repr__(self):
        s = "DistMatchLayer"
        return s