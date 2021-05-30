from collections import namedtuple
import numpy as np
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy.spatial import distance
import torch


# print(models.vgg16(pretrained=True).features)
# print(models.vgg19(pretrained=True).features)

class VGG16(nn.Module):
    def __init__(self, require_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.weights = [1, 1, 1, 1]
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        # print(vgg_pretrained_features)
        for i in range(4):
            self.slice1.add_module(str(i), vgg_pretrained_features[i])
        for i in range(4, 9):
            self.slice2.add_module(str(i), vgg_pretrained_features[i])
        for i in range(9, 16):
            self.slice3.add_module(str(i), vgg_pretrained_features[i])
        for i in range(16, 23):
            self.slice4.add_module(str(i), vgg_pretrained_features[i])
        if not require_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        h = self.slice1(input)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        out = vgg_outputs(h_relu1_2 * self.weights[0], h_relu2_2 * self.weights[1], h_relu3_3 * self.weights[2],
                          h_relu4_3 * self.weights[3])
        return out, h_relu2_2


class VGG19(nn.Module):
    def __init__(self, require_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.weights = [1, 1, 1, 1, 1]
        # print(vgg_pretrained_features)
        for i in range(4):
            self.slice1.add_module(str(i), vgg_pretrained_features[i])
        for i in range(4, 9):
            self.slice2.add_module(str(i), vgg_pretrained_features[i])
        for i in range(9, 16):
            self.slice3.add_module(str(i), vgg_pretrained_features[i])
        for i in range(16, 25):
            self.slice4.add_module(str(i), vgg_pretrained_features[i])
        for i in range(25, 36):
            self.slice5.add_module(str(i), vgg_pretrained_features[i])
        if not require_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        h = self.slice1(input)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_4 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_4"])
        out = vgg_outputs(h_relu1_2 * self.weights[0], h_relu2_2 * self.weights[1], h_relu3_3 * self.weights[2],
                          h_relu4_3 * self.weights[3], h_relu5_4 * self.weights[4])
        return out, h_relu2_2


class Mask_SSS(nn.Module):
    def __init__(self, init_value=[1], planes=None):
        super().__init__()
        self.planes = planes
        self.weight = Parameter(torch.Tensor(init_value))

    def forward(self, input):
        weight = self.weight
        if self.planes is not None:
            weight = self.weight[None, :, None, None]

        return input * weight


class Mask_FPGM:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            # print("filter codebook done")
        else:
            pass
        return codebook

    def get_filter_index(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            # print("filter index done")
        else:
            pass
        return filter_small_index, filter_large_index

    # optimize for fast ccalculation
    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length, dist_type="l2"):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)

            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]

            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            # for x1, x2 in enumerate(filter_large_index):
            #     for y1, y2 in enumerate(filter_large_index):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # distance using numpy function
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            # print('filter_large_index', filter_large_index)
            # print('filter_small_index', filter_small_index)
            # print('similar_sum', similar_sum)
            # print('similar_large_index', similar_large_index)
            # print('similar_small_index', similar_small_index)
            # print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            # print("similar index done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()
            # print(item.size())

        # for index, item in enumerate(self.model.named_parameters()):
        #     print(index, "->", item[0])

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
        # print(self.model_length)

    def init_rate(self, rate_norm_per_layer, rate_dist_per_layer, layer_begin, layer_end):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
            self.distance_rate[index] = 1
        for key in range(layer_begin, layer_end + 1):
            self.compress_rate[key] = rate_norm_per_layer
            self.distance_rate[key] = rate_dist_per_layer

        # self.mask_index = [x for x in range(layer_begin, layer_end + 1, 4)]
        self.mask_index = []
        for i in range(layer_begin, layer_end + 1, 4):
            self.mask_index.append(i)

        assert 0 <= layer_begin <= layer_end <= 184

    #        self.mask_index =  [x for x in range (0,330,3)]

    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer, dist_type, use_cuda=True, layer_begin=28,
                  layer_end=163):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer, layer_begin=layer_begin, layer_end=layer_end)
        # print(self.mask_index)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                # mask for norm criterion
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if use_cuda:
                    self.mat[index] = self.mat[index].cuda()

                # # get result about filter index
                # self.filter_small_index[index], self.filter_large_index[index] = \
                #     self.get_filter_index(item.data, self.compress_rate[index], self.model_length[index])

                # mask for distance criterion
                self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                     self.distance_rate[index],
                                                                     self.model_length[index], dist_type=dist_type)
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                if use_cuda:
                    self.similar_matrix[index] = self.similar_matrix[index].cuda()
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")

    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        print("mask similar Done")

    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                # reverse the mask of model
                # b = a * (1 - self.mat[index])
                b = a * self.mat[index]
                b = b * self.similar_matrix[index]
                item.grad.data = b.view(self.model_size[index])
        # print("grad zero Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))


def make_layer(block, num_block, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True,
               relu=True):
    layers = []
    for i in range(num_block):
        layers.append(
            block(in_channels, out_channels, kernel_size, stride=stride, upsample=upsample, normalize=normalize,
                  relu=relu))
    return nn.Sequential(*layers)


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 64, kernel_size=3, stride=1),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ConvBlock(128, 128, kernel_size=3, stride=1),
            ConvBlock(128, 256, kernel_size=3, stride=2),
            ConvBlock(256, 256, kernel_size=3, stride=1),
            make_layer(ResidualBlock, num_block=17, in_channels=256, out_channels=256, kernel_size=3, stride=1,
                       upsample=False, normalize=True, relu=True),
            ConvBlock(256, 256, kernel_size=3, upsample=False),
            ConvBlock(256, 128, kernel_size=3, upsample=True),
            ConvBlock(128, 128, kernel_size=3, upsample=False),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 64, kernel_size=3, upsample=False),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, upsample=False, normalize=False, relu=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class TransformerNet_Fire(nn.Module):
    def __init__(self):
        super(TransformerNet_Fire, self).__init__()
        self.model = nn.Sequential(
            ConvBlock_Fire(3, 32, kernel_size=9, stride=1),
            Fire(inplanes=32, squeeze_planes=8, expand1x1_planes=32, expand3x3_planes=32, stride=2),
            Fire(inplanes=64, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64, stride=2),
            Fire(inplanes=128, squeeze_planes=32, expand1x1_planes=128, expand3x3_planes=128, stride=2),
            make_layer(ResidualBlock, num_block=17, in_channels=256, out_channels=256, kernel_size=3, stride=1,
                       upsample=False, normalize=True, relu=True),
            Fire(inplanes=256, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64, upsample=True),
            Fire(inplanes=128, squeeze_planes=8, expand1x1_planes=32, expand3x3_planes=32, upsample=True),
            Fire(inplanes=64, squeeze_planes=4, expand1x1_planes=16, expand3x3_planes=16, upsample=True),
            ConvBlock_Fire(32, 3, kernel_size=9, upsample=False, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)


class TransformerNet_Scale(nn.Module):
    def __init__(self):
        super(TransformerNet_Scale, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 64, kernel_size=3, stride=1),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ConvBlock(128, 128, kernel_size=3, stride=1),
            ConvBlock(128, 256, kernel_size=3, stride=2),
            ConvBlock(256, 256, kernel_size=3, stride=1),
            make_layer(SparseResBasicBlock, num_block=17, in_channels=256, out_channels=256, kernel_size=3, stride=1,
                       upsample=False, normalize=True, relu=True),
            ConvBlock(256, 256, kernel_size=3, upsample=False),
            ConvBlock(256, 128, kernel_size=3, upsample=True),
            ConvBlock(128, 128, kernel_size=3, upsample=False),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 64, kernel_size=3, upsample=False),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, upsample=False, normalize=False, relu=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class TransformerNet_Scale_Fire(nn.Module):
    def __init__(self):
        super(TransformerNet_Scale_Fire, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            Fire(inplanes=32, squeeze_planes=8, expand1x1_planes=32, expand3x3_planes=32, stride=2),
            Fire(inplanes=64, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64, stride=2),
            Fire(inplanes=128, squeeze_planes=32, expand1x1_planes=128, expand3x3_planes=128, stride=2),
            make_layer(SparseResBasicBlock, num_block=17, in_channels=256, out_channels=256, kernel_size=3, stride=1,
                       upsample=False, normalize=True, relu=True),
            Fire(inplanes=256, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64, upsample=True),
            Fire(inplanes=128, squeeze_planes=8, expand1x1_planes=32, expand3x3_planes=32, upsample=True),
            Fire(inplanes=64, squeeze_planes=4, expand1x1_planes=16, expand3x3_planes=16, upsample=True),
            ConvBlock(32, 3, kernel_size=9, upsample=False, normalize=False, relu=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class TransformerNet_Complex(nn.Module):
    def __init__(self, num_block=7):
        super(TransformerNet_Complex, self).__init__()
        self.model = nn.Sequential(
            ConvBlock_Fire(3, 32-3-3, kernel_size=9, stride=1),
            Fire(inplanes=32-3-3, squeeze_planes=8, expand1x1_planes=32-6, expand3x3_planes=32-6, stride=2),
            Fire(inplanes=64-6-6, squeeze_planes=16, expand1x1_planes=64-12, expand3x3_planes=64-12, stride=2),
            Fire(inplanes=128-12-12, squeeze_planes=32, expand1x1_planes=128-25, expand3x3_planes=128-25, stride=2),
            make_layer(ResidualBlock, num_block=num_block, in_channels=256-25-25, out_channels=256-25-25, kernel_size=3, stride=1,
                       upsample=False, normalize=True, relu=True),
            Fire(inplanes=256-25-25, squeeze_planes=16, expand1x1_planes=64-12, expand3x3_planes=64-12, upsample=True),
            Fire(inplanes=128-12-12, squeeze_planes=8, expand1x1_planes=32-6, expand3x3_planes=32-6, upsample=True),
            Fire(inplanes=64-6-6, squeeze_planes=4, expand1x1_planes=16-3, expand3x3_planes=16-3, upsample=True),
            ConvBlock_Fire(32-3-3, 3, kernel_size=9, upsample=False, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=False, normalize=True,
                 relu=True):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, normalize=normalize,
                      relu=relu),
            ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, normalize=normalize,
                      relu=False)
        )

    def forward(self, x):
        return self.block(x) + x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        # 一个channel内做归一化，算H*W的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        # print(x.size())
        return x


class ConvBlock_Fire(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True,
                 padding=0):
        super(ConvBlock_Fire, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )
        # 一个channel内做归一化，算H*W的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        # print(x.size())
        return x


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, stride=1, upsample=False):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = ConvBlock_Fire(inplanes, squeeze_planes, kernel_size=1)
        self.expand1x1 = ConvBlock_Fire(squeeze_planes, expand1x1_planes, stride=stride, kernel_size=1,
                                        upsample=upsample)
        self.expand3x3 = ConvBlock_Fire(squeeze_planes, expand3x3_planes, stride=stride, kernel_size=3,
                                        upsample=upsample)

    def forward(self, x):
        # print("in fire")
        x = self.squeeze(x)
        x = torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1)
        # print(x.size())
        return x


class SparseResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=False, normalize=True,
                 relu=True):
        super(SparseResBasicBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, normalize=normalize,
                      relu=relu),
            ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, normalize=normalize,
                      relu=False)
        )
        m = [1]
        self.mask = Mask_SSS(m)

    def forward(self, x):
        return self.mask(self.block(x)) + x
