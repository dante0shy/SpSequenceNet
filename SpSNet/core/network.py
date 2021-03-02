import sparseconvnet as scn
from SpSNet.utils.SparseGlobalPooling import GlobalMeanAttentionPooling
from SpSNet.utils.GlobalPoolLayer import GlobalMaskLayer
from SpSNet.utils.SubstractLayer import DistMatchLayer_v4
import torch.nn as nn
import torch
import time

# m = 16

class Model(nn.Module):
    leakiness = 0
    downsample = [2, 2]

    def __init__(self,class_num, full_scale ,m = 16,dimension=3):
        nn.Module.__init__(self)
        self.dimension = dimension
        self.input = scn.InputLayer(dimension, full_scale, mode=4)
        self.input_s = scn.InputLayer(dimension, full_scale, mode=4)
        self.down_in = scn.SubmanifoldConvolution(dimension, 1, m, 3, False)
        self.down_in_s = scn.SubmanifoldConvolution(dimension, 1, m, 3, False)
        self.main_block1 = self.block(m, m, 2, 1)
        self.main_block2 = self.block(m, 2 * m, 1, 2)
        self.main_block3 = self.block(2 * m, 3 * m, 1, 2)
        self.main_block4 = self.block(3 * m, 4 * m, 1, 2)
        self.main_block5 = self.block(4 * m, 5 * m, 1, 2)
        self.main_block6 = self.block(5 * m, 6 * m, 1, 2)
        self.main_block7 = self.block(6 * m, 7 * m, 2, 2)
        self.main_block8 = self.block(7 * m, 7 * m, 2, 1)

        self.support_block1 = self.block(m, m, 2, 1)
        self.support_block2 = self.block(m, 2 * m, 1, 2)
        self.support_block3 = self.block(2 * m, 3 * m, 1, 2)
        self.support_block4 = self.block(3 * m, 4 * m, 1, 2)
        self.support_block5 = self.block(4 * m, 5 * m, 1, 2)
        self.support_block6 = self.block(5 * m, 6 * m, 1, 2)
        self.support_block7 = self.block(6 * m, 7 * m, 1, 2)
        self.support_block8 = self.block(7 * m, 7 * m, 1, 1)

        self.support_block2_tune = self.guide_tune(
            dimension, 2 * m, 2 * m, 1, False
        )
        self.support_block2_out = GlobalMeanAttentionPooling(dimension)
        self.support_block3_tune = self.guide_tune(
            dimension, 4 * m, 4 * m, 1, False
        )
        self.support_block3_out = GlobalMeanAttentionPooling(dimension)
        self.support_block4_tune = self.guide_tune(
            dimension, 7 * m, 7 * m, 1, False
        )
        self.support_block4_out = GlobalMeanAttentionPooling(dimension)

        self.global_add2 = GlobalMaskLayer(dimension)
        self.global_add3 = GlobalMaskLayer(dimension)
        self.global_add4 = GlobalMaskLayer(dimension)

        self.spatial_pick = DistMatchLayer_v4(dimension, 7 * m,topk=3)
        # self.join_sub = scn.JoinTable()
        self.tune_sub = self.guide_tune(dimension, 14 * m, 7 * m, 1, False)

        self.deconv7 = self.decoder(7 * m, 6 * m)
        self.join6 = scn.JoinTable()
        self.deconv6 = self.decoder(12 * m, 5 * m)
        self.deconv5 = self.decoder(5 * m, 4 * m)
        self.join4 = scn.JoinTable()
        self.deconv4 = self.decoder(8 * m, 3 * m)
        self.deconv3 = self.decoder(3 * m, 2 * m)
        self.join2 = scn.JoinTable()
        self.deconv2 = self.decoder(4 * m, 2 * m)
        # self.deconv1 = self.decoder(2 * m, m)

        self.output = scn.OutputLayer(dimension)
        self.linear = nn.Linear(2 * m, class_num)

    def support_feature(self, x):
        time_s = time.time()
        down_feature = self.down_in_s(x)
        support_1 = self.support_block1(down_feature)
        support_2 = self.support_block2(support_1)
        support_3 = self.support_block3(support_2)
        support_4 = self.support_block4(support_3)
        support_5 = self.support_block5(support_4)
        support_6 = self.support_block6(support_5)
        support_7 = self.support_block7(support_6)
        support_8 = self.support_block8(support_7)
        support_2m_f = self.support_block2_tune(support_2)
        support_2m_f = self.support_block2_out(support_2m_f)
        support_4m_f = self.support_block3_tune(support_4)
        support_4m_f = self.support_block3_out(support_4m_f)
        support_8m_f = self.support_block4_tune(support_7)
        support_8m_f = self.support_block4_out(support_8m_f)
        return support_2m_f, support_4m_f, support_8m_f, support_8

    def forward(self, x):
        start_time = time.time()
        support_frames = []
        for i in range(1, len(x[0])):
            support_frames.append([x[0][i], x[1][i]])
        # for support_frame in support_frames:
        pre_frame = [x[0][0], x[1][0]]
        x = self.input(pre_frame)
        x = self.down_in(x)
        x = self.main_block1(x)
        x = self.main_block2(x)
        mt1, mt2, mt3, mt4 = self.support_feature(self.input_s(support_frames[0]))
        feature_2 = self.global_add2(x, mt1)
        feature_3 = self.main_block3(feature_2)
        feature_4 = self.main_block4(feature_3)
        feature_4 = self.global_add3(feature_4, mt2)
        feature_5 = self.main_block5(feature_4)
        feature_6 = self.main_block6(feature_5)
        feature_7 = self.main_block7(feature_6)
        feature_7 = self.global_add4(feature_7, mt3)
        feature_8 = self.main_block8(feature_7)
        feature_sub = self.spatial_pick(feature_8, mt4)
        feature_8 = self.tune_sub(feature_sub)

        decoder_7 = self.deconv7(feature_8)
        decoder_6 = self.join6([decoder_7, feature_6])
        decoder_6 = self.deconv6(decoder_6)
        decoder_5 = self.deconv5(decoder_6)
        decoder_4 = self.join4([decoder_5, feature_4])
        decoder_4 = self.deconv4(decoder_4)
        decoder_3 = self.deconv3(decoder_4)
        decoder_2 = self.join2([decoder_3, feature_2])
        decoder_2 = self.deconv2(decoder_2)
        # decoder_1 = self.deconv1(decoder_2)
        out = self.output(decoder_2)
        out = self.linear(out)
        return out

    def guide_tune(self, dimension, a, b, fs, leakiness=0.):
        return scn.Sequential() \
            .add(scn.SubmanifoldConvolution(dimension, a, b, fs, False)) \
            .add(scn.BatchNormLeakyReLU(b, leakiness=leakiness)) \
            .add(scn.SubmanifoldConvolution(dimension, b, b, fs, False))

    def decoder(self, a, b):
        return (
            scn.Sequential()
                .add(scn.BatchNormLeakyReLU(a))
                .add(scn.Deconvolution(self.dimension, a, b, 2, 2, False))
                .add(self.decoder_block(b, b, 1, 1))
        )

    def residual(self, nIn, nOut, stride):
        if stride > 1:
            return scn.Convolution(self.dimension, nIn, nOut, 2, stride, False)
        elif nIn != nOut:
            return scn.NetworkInNetwork(nIn, nOut, False)
        else:
            return scn.Identity()

    def decoder_block(self, nPlanes, n, reps, stride):
        m = scn.Sequential()
        for rep in range(reps):
            m.add(
                scn.ConcatTable()
                    .add(
                    scn.Sequential()
                        .add(scn.BatchNormReLU(nPlanes))
                        .add(scn.SubmanifoldConvolution(self.dimension, nPlanes, n, 3, False))
                    # .add(scn.BatchNormReLU(n))
                    # .add(scn.SubmanifoldConvolution(dimension, n, n, 3, False))
                )
                    .add(scn.Identity())
            )
            m.add(scn.AddTable())
            nPlanes = n
        return m

    def block(self, nPlanes, n, reps, stride):
        m = scn.Sequential()
        for rep in range(reps):
            if rep == 0:
                m.add(scn.BatchNormReLU(nPlanes))
                m.add(
                    scn.ConcatTable()
                        .add(self.residual(nPlanes, n, stride))
                        .add(
                        scn.Sequential()
                            .add(
                            scn.SubmanifoldConvolution(self.dimension, nPlanes, n, 3, False)
                            if stride == 1
                            else scn.Convolution(
                                self.dimension, nPlanes, n, 2, stride, False
                            )
                        )
                            .add(scn.BatchNormReLU(n))
                            .add(scn.SubmanifoldConvolution(self.dimension, n, n, 3, False))
                    )
                )
            else:
                m.add(
                    scn.ConcatTable()
                        .add(
                        scn.Sequential()
                            .add(scn.BatchNormReLU(nPlanes))
                            .add(
                            scn.SubmanifoldConvolution(self.dimension, nPlanes, n, 3, False)
                        )
                            .add(scn.BatchNormReLU(n))
                            .add(scn.SubmanifoldConvolution(self.dimension, n, n, 3, False))
                    )
                        .add(scn.Identity())
                )
            m.add(scn.AddTable())
            nPlanes = n
        return m


if __name__ == "__main__":
    a = Model()
