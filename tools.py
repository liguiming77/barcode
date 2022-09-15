# -*- coding: utf-8 -*-
from PIL import Image,ImageChops,ImageEnhance
import numpy as np
import random
import torch
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.transforms import RandomResizedCrop
import cv2
from torch.nn import Conv2d
from torch import nn
from torchvision import datasets, transforms,models

# filter_size_w_h_s = (640,480,0.05)
# name2tag={'208': 0, '402': 1, '403': 2, '其他': 3, '其他塑料': 4, '其他织物': 5, '其他金属': 6, '其他非回收': 7, '医疗废品': 8, '危险物品': 9, '可回收物': 10, '小家电': 11, '干湿垃圾': 12, '建筑垃圾': 13, '感光度低': 14, '摄像头装反': 15, '易拉罐': 16, '有害垃圾': 17, '玻璃制品': 18, '疑似非回收': 19, '碎玻璃': 20, '纸类': 21, '绿屏': 22, '衣服': 23, '角度需调整': 24, '违法物品': 25, '鞋子': 26, '饮料瓶': 27, '黑屏': 28}
num_classes = 2


newsize=(50,50) # w,h
def default_loader(path,is_observ=False):
    img_pil =  Image.open(path).convert("RGB")
    if not is_observ:
        img_pil = img_pil.crop( (0,0,img_pil.width, int( (img_pil.height*2)/3 )  )  )
    # img_pil = Image.open(path).convert("L")
    # img_np = np.asarray(img_pil)
    # img_np = np.resize(img_np,newsize)
    # img_np = np.expand_dims(img_np,-1) ## 增加一个纬度
    # if is_observ:
    #     img_pil=img_pil.transpose(Image.ROTATE_270) ## totate 270
    # img_pil = img_pil.resize(newsize)

    return img_pil

def base_loader(path,w=None,h=None,is_observ=False):

    if '.png' in path:
        img_pil = default_loader(path,is_observ=is_observ)
    elif '.yuv' in path:
        img_pil = decode_yuv(path,width=w,height=h)
    img_pil = img_pil.resize(newsize)
    return img_pil

from typing import Callable, List, Optional
class newConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [torch.nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  dilation=dilation, groups=groups, bias=False)]
        layers.append( torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  dilation=dilation, groups=groups, bias=False) )
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels

class SigmodLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SigmodLinear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        return self.sigmod(self.linear(x)) ##x.shape [64, 1280]

class newmobilenetv2(nn.Module):
    def __init__(self, in_features, out_features,pretrained=True):
        super(newmobilenetv2, self).__init__()
        self.net = models.mobilenet_v2(pretrained=pretrained)
        self.net.classifier[1] = SigmodLinear(in_features, out_features)
        self.net.features[0][0] = Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # self.net.features[0] = newConvNormActivation(in_channels=6,out_channels=32,kernel_size=(3, 3), stride=(2, 2),padding=(1, 1))

    def forward(self, x1,x2):
        x = torch.cat((x1,x2),dim=1)
        return self.net(x) ##x.shape [64, 1280]
# class newmobilenetv2(nn.Module):
#     def __init__(self, in_features, out_features,pretrained=True):
#         super(newmobilenetv2, self).__init__()
#         self.net = models.mobilenet_v2(pretrained=pretrained)
#         self.net.classifier[1] = SigmodLinear(in_features, out_features)
#         self.net.features[0][0] = Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     def forward(self, x1,x2):
#         x = torch.cat((x1,x2),dim=1)
#         return self.net(x) ##x.shape [64, 1280]
class newefficientnet(nn.Module):
    def __init__(self, in_features, out_features):
        super(newefficientnet, self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b5',in_channels=6)
        self.net._fc = SigmodLinear(in_features, out_features)

    def forward(self, x1,x2):
        x = torch.cat((x1,x2),dim=1)
        return self.net(x)


class LocalAgg(nn.Module):
    """
    局部模块，LocalAgg
    卷积操作能够有效的提取局部特征
    为了能够降低计算量，使用 逐点卷积+深度可分离卷积实现
    """

    def __init__(self, channels):
        super(LocalAgg, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        # 逐点卷积，相当于全连接层。增加非线性，提高特征提取能力
        self.pointwise_conv_0 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # 深度可分离卷积
        self.depthwise_conv = nn.Conv2d(channels, channels, padding=1, kernel_size=3, groups=channels, bias=False)
        # 归一化
        self.pointwise_prenorm_1 = nn.BatchNorm2d(channels)
        # 逐点卷积，相当于全连接层，增加非线性，提高特征提取能力
        self.pointwise_conv_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_prenorm_1(x)
        x = self.pointwise_conv_1(x)
        return x

class GlobalSparseAttention(nn.Module):
    """
    全局模块，选取特定的tokens,进行全局作用
    """

    def __init__(self, channels, r, heads):
        """

        Args:
            channels: 通道数
            r: 下采样倍率
            heads: 注意力头的数目
                   这里使用的是多头注意力机制，MHSA,multi-head self-attention
        """
        super(GlobalSparseAttention, self).__init__()
        #
        self.head_dim = channels // heads
        # 扩张的
        self.scale = self.head_dim ** -0.5

        self.num_heads = heads

        # 使用平均池化,来进行特征提取
        self.sparse_sampler = nn.AvgPool2d(kernel_size=1, stride=r)

        # 计算qkv
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, self.num_heads, -1, H * W).split([self.head_dim,self.head_dim,self.head_dim],dim=2)
        # 计算特征图 attention map
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        # value和特征图进行计算，得出全局注意力的结果
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        # print(x.shape)
        return x

class LocalPropagation(nn.Module):
    def __init__(self, channels, r):
        super(LocalPropagation, self).__init__()

        # 组归一化
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        # 使用转置卷积 恢复 GlobalSparseAttention模块 r倍的下采样率
        self.local_prop = nn.ConvTranspose2d(channels,
                                             channels,
                                             kernel_size=r,
                                             stride=r,
                                             groups=channels)
        # 使用逐点卷积
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.local_prop(x)
        x = self.norm(x)
        x = self.proj(x)
        return x

import torch
import torch.nn as nn


class Residual(nn.Module):
    """
    残差网络
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class ConditionalPositionalEncoding(nn.Module):
    """
    条件编码信息
    """

    def __init__(self, channels):
        super(ConditionalPositionalEncoding, self).__init__()
        self.conditional_positional_encoding = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                                                         bias=False)

    def forward(self, x):
        x = self.conditional_positional_encoding(x)
        return x


class MLP(nn.Module):
    """
    FFN 模块
    """

    def __init__(self, channels):
        super(MLP, self).__init__()
        expansion = 4
        self.mlp_layer_0 = nn.Conv2d(channels, channels * expansion, kernel_size=1, bias=False)
        self.mlp_act = nn.GELU()
        self.mlp_layer_1 = nn.Conv2d(channels * expansion, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.mlp_layer_0(x)
        x = self.mlp_act(x)
        x = self.mlp_layer_1(x)
        return x


class LocalAgg(nn.Module):
    """
    局部模块，LocalAgg
    卷积操作能够有效的提取局部特征
    为了能够降低计算量，使用 逐点卷积+深度可分离卷积实现
    """

    def __init__(self, channels):
        super(LocalAgg, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        # 逐点卷积，相当于全连接层。增加非线性，提高特征提取能力
        self.pointwise_conv_0 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # 深度可分离卷积
        self.depthwise_conv = nn.Conv2d(channels, channels, padding=1, kernel_size=3, groups=channels, bias=False)
        # 归一化
        self.pointwise_prenorm_1 = nn.BatchNorm2d(channels)
        # 逐点卷积，相当于全连接层，增加非线性，提高特征提取能力
        self.pointwise_conv_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_prenorm_1(x)
        x = self.pointwise_conv_1(x)
        return x


class GlobalSparseAttention(nn.Module):
    """
    全局模块，选取特定的tokens,进行全局作用
    """

    def __init__(self, channels, r, heads):
        """

        Args:
            channels: 通道数
            r: 下采样倍率
            heads: 注意力头的数目
                   这里使用的是多头注意力机制，MHSA,multi-head self-attention
        """
        super(GlobalSparseAttention, self).__init__()
        #
        self.head_dim = channels // heads
        # 扩张的
        self.scale = self.head_dim ** -0.5

        self.num_heads = heads

        # 使用平均池化,来进行特征提取
        self.sparse_sampler = nn.AvgPool2d(kernel_size=1, stride=r)

        # 计算qkv
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, self.num_heads, -1, H * W).split([self.head_dim, self.head_dim, self.head_dim],
                                                                       dim=2)
        # 计算特征图 attention map
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        # value和特征图进行计算，得出全局注意力的结果
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        # print(x.shape)
        return x


class LocalPropagation(nn.Module):
    def __init__(self, channels, r):
        super(LocalPropagation, self).__init__()

        # 组归一化
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        # 使用转置卷积 恢复 GlobalSparseAttention模块 r倍的下采样率
        self.local_prop = nn.ConvTranspose2d(channels,
                                             channels,
                                             kernel_size=r,
                                             stride=r,
                                             groups=channels)
        # 使用逐点卷积
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.local_prop(x)
        x = self.norm(x)
        x = self.proj(x)
        return x


class LGL(nn.Module):
    def __init__(self, channels, r, heads):
        super(LGL, self).__init__()

        self.cpe1 = ConditionalPositionalEncoding(channels)
        self.LocalAgg = LocalAgg(channels)
        self.mlp1 = MLP(channels)
        self.cpe2 = ConditionalPositionalEncoding(channels)
        self.GlobalSparseAttention = GlobalSparseAttention(channels, r, heads)
        self.LocalPropagation = LocalPropagation(channels, r)
        self.mlp2 = MLP(channels)

    def forward(self, x):
        # 1. 经过 位置编码操作
        x = self.cpe1(x) + x
        # 2. 经过第一步的 局部操作
        x = self.LocalAgg(x) + x
        # 3. 经过一个前馈网络
        x = self.mlp1(x) + x
        # 4. 经过一个位置编码操作
        x = self.cpe2(x) + x
        # 5. 经过一个全局捕捉的操作。长和宽缩小 r倍。然后通过一个
        # 6. 经过一个 局部操作部
        x = self.LocalPropagation(self.GlobalSparseAttention(x)) + x
        # 7. 经过一个前馈网络
        x = self.mlp2(x) + x

        return x


# if __name__ == '__main__':
#     # 64通道，图片大小为32*32
#     x = torch.randn(size=(1, 64, 32, 32))
#     # 64通道，下采样2倍，8个头的注意力
#     model = LGL(64, 2, 8)
#     out = model(x)
#     print(out.shape)



# edgevits的配置信息
edgevit_configs = {
    'XXS': {
        'channels': (36, 72, 144, 288),
        'blocks': (1, 1, 3, 2),
        'heads': (1, 2, 4, 8)
    }
    ,
    'XS': {
        'channels': (48, 96, 240, 384),
        'blocks': (1, 1, 2, 2),
        'heads': (1, 2, 4, 8)
    }
    ,
    'S': {
        'channels': (48, 96, 240, 384),
        'blocks': (1, 2, 3, 2),
        'heads': (1, 2, 4, 8)
    }
}

HYPERPARAMETERS = {
    'r': (4, 2, 2, 1)
}


class Residual(nn.Module):
    """
    残差网络
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class ConditionalPositionalEncoding(nn.Module):
    """

    """

    def __init__(self, channels):
        super(ConditionalPositionalEncoding, self).__init__()
        self.conditional_positional_encoding = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                                                         bias=False)

    def forward(self, x):
        x = self.conditional_positional_encoding(x)
        return x


class MLP(nn.Module):
    """
    FFN 模块
    """

    def __init__(self, channels):
        super(MLP, self).__init__()
        expansion = 4
        self.mlp_layer_0 = nn.Conv2d(channels, channels * expansion, kernel_size=1, bias=False)
        self.mlp_act = nn.GELU()
        self.mlp_layer_1 = nn.Conv2d(channels * expansion, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.mlp_layer_0(x)
        x = self.mlp_act(x)
        x = self.mlp_layer_1(x)
        return x


class LocalAgg(nn.Module):
    """
    局部模块，LocalAgg
    卷积操作能够有效的提取局部特征
    为了能够降低计算量，使用 逐点卷积+深度可分离卷积实现
    """

    def __init__(self, channels):
        super(LocalAgg, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        # 逐点卷积，相当于全连接层。增加非线性，提高特征提取能力
        self.pointwise_conv_0 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # 深度可分离卷积
        self.depthwise_conv = nn.Conv2d(channels, channels, padding=1, kernel_size=3, groups=channels, bias=False)
        # 归一化
        self.pointwise_prenorm_1 = nn.BatchNorm2d(channels)
        # 逐点卷积，相当于全连接层，增加非线性，提高特征提取能力
        self.pointwise_conv_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_prenorm_1(x)
        x = self.pointwise_conv_1(x)
        return x


class GlobalSparseAttention(nn.Module):
    """
    全局模块，选取特定的tokens,进行全局作用
    """

    def __init__(self, channels, r, heads):
        """

        Args:
            channels: 通道数
            r: 下采样倍率
            heads: 注意力头的数目
                   这里使用的是多头注意力机制，MHSA,multi-head self-attention
        """
        super(GlobalSparseAttention, self).__init__()
        #
        self.head_dim = channels // heads
        # 扩张的
        self.scale = self.head_dim ** -0.5

        self.num_heads = heads

        # 使用平均池化,来进行特征提取
        self.sparse_sampler = nn.AvgPool2d(kernel_size=1, stride=r)

        # 计算qkv
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, self.num_heads, -1, H * W).split([self.head_dim, self.head_dim, self.head_dim],
                                                                       dim=2)
        # 计算特征图 attention map
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        # value和特征图进行计算，得出全局注意力的结果
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        # print(x.shape)
        return x


class LocalPropagation(nn.Module):
    def __init__(self, channels, r):
        super(LocalPropagation, self).__init__()

        # 组归一化
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        # 使用转置卷积 恢复 GlobalSparseAttention模块 r倍的下采样率
        self.local_prop = nn.ConvTranspose2d(channels,
                                             channels,
                                             kernel_size=r,
                                             stride=r,
                                             groups=channels)
        # 使用逐点卷积
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.local_prop(x)
        x = self.norm(x)
        x = self.proj(x)
        return x


class LGL(nn.Module):
    def __init__(self, channels, r, heads):
        super(LGL, self).__init__()

        self.cpe1 = ConditionalPositionalEncoding(channels)
        self.LocalAgg = LocalAgg(channels)
        self.mlp1 = MLP(channels)
        self.cpe2 = ConditionalPositionalEncoding(channels)
        self.GlobalSparseAttention = GlobalSparseAttention(channels, r, heads)
        self.LocalPropagation = LocalPropagation(channels, r)
        self.mlp2 = MLP(channels)

    def forward(self, x):
        # 1. 经过 位置编码操作
        x = self.cpe1(x) + x
        # 2. 经过第一步的 局部操作
        x = self.LocalAgg(x) + x
        # 3. 经过一个前馈网络
        x = self.mlp1(x) + x
        # 4. 经过一个位置编码操作
        x = self.cpe2(x) + x
        # 5. 经过一个全局捕捉的操作。长和宽缩小 r倍。然后通过一个
        # 6. 经过一个 局部操作部
        x = self.LocalPropagation(self.GlobalSparseAttention(x)) + x
        # 7. 经过一个前馈网络
        x = self.mlp2(x) + x

        return x


class DownSampleLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(DownSampleLayer, self).__init__()
        self.downsample = nn.Conv2d(dim_in,
                                    dim_out,
                                    kernel_size=r,
                                    stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.downsample(x)
        x = self.norm(x)
        return x

    # if __name__ == '__main__':


#     # 64通道，图片大小为32*32
#     x = torch.randn(size=(1, 64, 32, 32))
#     # 64通道，下采样2倍，8个头的注意力
#     model = LGL(64, 2, 8)
#     out = model(x)
#     print(out.shape)


class EdgeViT(nn.Module):

    def __init__(self, channels, blocks, heads,inchannels, r=[4, 2, 2, 1], num_classes=1000, distillation=False):
        super(EdgeViT, self).__init__()
        self.distillation = distillation
        l = []
        in_channels = inchannels
        # 主体部分
        for stage_id, (num_channels, num_blocks, num_heads, sample_ratio) in enumerate(zip(channels, blocks, heads, r)):
            # print(num_channels,num_blocks,num_heads,sample_ratio)
            # print(in_channels)
            l.append(DownSampleLayer(dim_in=in_channels, dim_out=num_channels, r=4 if stage_id == 0 else 2))
            for _ in range(num_blocks):
                l.append(LGL(channels=num_channels, r=sample_ratio, heads=num_heads))

            in_channels = num_channels

        self.main_body = nn.Sequential(*l)
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(in_channels, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()


        if self.distillation:
            self.dist_classifier = nn.Linear(in_channels, num_classes, bias=True)
        # print(self.main_body)

    def forward(self, x1,x2):
        # print(x.shape)
        x = torch.cat((x1, x2), dim=1)
        x = self.main_body(x)
        x = self.pooling(x).flatten(1)

        if self.distillation:
            x = self.classifier(x), self.dist_classifier(x)

            if not self.training:
                x = 1 / 2 * (x[0] + x[1])
        else:
            x =  self.sigmoid(self.classifier(x))

        return x


def EdgeViT_XXS(pretrained=False):
    model = EdgeViT(**edgevit_configs['XXS'])

    if pretrained:
        raise NotImplementedError

    return model


def EdgeViT_XS(pretrained=False):
    model = EdgeViT(**edgevit_configs['XS'])

    if pretrained:
        raise NotImplementedError

    return model


def EdgeViT_S(pretrained=False):
    model = EdgeViT(**edgevit_configs['S'])

    if pretrained:
        raise NotImplementedError

    return model


# crop_transform = RandomResizedCrop(size=(newsize[1],newsize[0]),scale=(0.005,0.8),ratio=(0.1,10) )
#
# def crop(pil_img, zone):#(x1,y1,x2,y2)
#     img = pil_img.copy()
#     left, upper, right, lower = zone
#     """
#         所截区域图片保存
#     :param path: 图片路径
#     :param left: 区块左上角位置的像素点离图片左边界的距离
#     :param upper：区块左上角位置的像素点离图片上边界的距离
#     :param right：区块右下角位置的像素点离图片左边界的距离
#     :param lower：区块右下角位置的像素点离图片上边界的距离
#      故需满足：lower > upper、right > left
#     :param save_path: 所截图片保存位置
#     """
#     # img = Image.open(path)  # 打开图像
#     box = (left, upper, right, lower)
#     roi = img.crop(box)
#     return roi
#
# def numpy2cv(shape=(10,10)): ## shape = cvimg[0:2]
#     img = np.ones(shape)
#     img = np.float32(img)
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     return img
#
# def bytes2cv(im):
#     '''二进制图片转cv2
#
#     :param im: 二进制图片数据，bytes
#     :return: cv2图像，numpy.ndarray
#     '''
#     return cv2.imdecode(np.array(bytearray(im), dtype='uint8'), cv2.IMREAD_COLOR)  # 从二进制图片数据中读取
#
#
# def cv2bytes(im,ext='.png'):
#     '''cv2转二进制图片
#
#     :param im: cv2图像，numpy.ndarray
#     :return: 二进制图片数据，bytes
#     '''
#     return np.array(cv2.imencode(ext, im)[1]).tobytes()
#
#
# def pil2cv(img):
#     #img = Image.open(path).convert("RGB")#.convert("RGB")可不要，默认打开就是RGB
#     img = img.convert("RGB")
#     img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
#     return img
#
# def cv2pil(img):
#     return  Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# #scale:指定裁剪的随机区域的下限和上限  ratio:宽高比的上下限
# class CustomRandomResizedCrop(RandomResizedCrop):
#     """Crop a random portion of image and resize it to a given size.
#
#     If the image is torch Tensor, it is expected
#     to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
#
#     A crop of the original image is made: the crop has a random area (H * W)
#     and a random aspect ratio. This crop is finally resized to the given
#     size. This is popularly used to train the Inception networks.
#
#     Args:
#         size (int or sequence): expected output size of the crop, for each edge. If size is an
#             int instead of sequence like (h, w), a square output size ``(size, size)`` is
#             made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
#
#             .. note::
#                 In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
#         scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
#             before resizing. The scale is defined with respect to the area of the original image.
#         ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
#             resizing.
#         interpolation (InterpolationMode): Desired interpolation enum defined by
#             :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
#             If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
#             ``InterpolationMode.BICUBIC`` are supported.
#             For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
#
#     """
#     def custom_forward(self, img1,img2):##img1 different pic; img2 same pic
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be cropped and resized.
#
#         Returns:
#             PIL Image or Tensor: Randomly cropped and resized image.
#         """
#         i, j, h, w = self.get_params(img1, self.scale, self.ratio)
#         # while not (j>filter_size_w_h_s[0]*filter_size_w_h_s[2] and i> filter_size_w_h_s[1]*filter_size_w_h_s[2] and \
#         #         filter_size_w_h_s[0] - (j+w)>filter_size_w_h_s[0]*filter_size_w_h_s[2] and \
#         #     filter_size_w_h_s[1] - (i+h)>filter_size_w_h_s[1]*filter_size_w_h_s[2]):
#
#         # x1,y1,x2,y2   0.06 0.02 0.08
#         while not (j > filter_size_w_h_s[0] * (filter_size_w_h_s[2] +0.02 ) and  \
#                    i > filter_size_w_h_s[1] * (filter_size_w_h_s[2]-0.02)   and  \
#                    filter_size_w_h_s[0] - (j + w) > filter_size_w_h_s[0] * (filter_size_w_h_s[2]+0.02) and \
#                    filter_size_w_h_s[1] - (i + h) > filter_size_w_h_s[1] * filter_size_w_h_s[2] ):
#             i, j, h, w = self.get_params(img1, self.scale, self.ratio)
#         # crop_img1=crop(img1,[j,i,j+w,i+h])
#         crop_img2_1 = crop(img2,[j,i,j+w,i+h])
#         i2, j2, h2, w2 = self.get_params(crop_img2_1, self.scale, self.ratio)
#         crop_img2_2 = crop(crop_img2_1, [j2,i2,j2+w2,i2+h2])
#         wrand = random.randint( 0, w - w2)
#         hrand = random.randint(0, h - h2)
#         img1.paste(crop_img2_2, ( j+wrand,i+hrand ))
#         return F.resized_crop(img1, i, j, h, w, self.size, self.interpolation),torch.tensor([i, j, h, w]) ## y1,x1,h,w
#     def forward(self, img):
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be cropped and resized.
#
#         Returns:
#             PIL Image or Tensor: Randomly cropped and resized image.
#         """
#         i, j, h, w = self.get_params(img, self.scale, self.ratio)
#         while not (j > filter_size_w_h_s[0] * (filter_size_w_h_s[2] +0.02 ) and  \
#                    i > filter_size_w_h_s[1] * (filter_size_w_h_s[2]-0.02)   and  \
#                    filter_size_w_h_s[0] - (j + w) > filter_size_w_h_s[0] * (filter_size_w_h_s[2]+0.02) and \
#                    filter_size_w_h_s[1] - (i + h) > filter_size_w_h_s[1] * filter_size_w_h_s[2] ):
#             i, j, h, w = self.get_params(img, self.scale, self.ratio)
#         return F.resized_crop(img, i, j, h, w, self.size, self.interpolation),torch.tensor([i, j, h, w]) ## y1,x1,h,w
#

def image_enhanced(img):
    """图像随机扩充"""
    rand_choice = random.randrange(0, 2)   # 随机左右翻转
    rand = random.randrange(-60, 60, 10)    # 选择旋转角度-60~60（逆时针旋转）
    randbri = random.choice([0.6, 0.8, 1.0, 1.2,1.4])   # 选择亮度，大于1增强，小于1减弱
    randcol = random.choice([0.7, 0.9, 1.0,1.1, 1.3])   # 选择色度，大于1增强，小于1减弱
    randcon = random.choice([0.7, 0.9, 1.0,1.1, 1.3])   # 选择对比度，大于1增强，小于1减弱
    randsha = random.choice([0.5, 1.0, 1.5])        # 选择锐度，大于1增强，小于1减弱
    if rand_choice == 0:
        lr = img.transpose(Image.FLIP_LEFT_RIGHT)
        out1=lr.rotate(rand)
        bri = ImageEnhance.Brightness(out1)
        bri_img1 = bri.enhance(randbri)
        col = ImageEnhance.Color(bri_img1)
        col_img1 = col.enhance(randcol)
        con = ImageEnhance.Contrast(col_img1)
        con_img1 = con.enhance(randcon)
        sha = ImageEnhance.Sharpness(con_img1)
        sha_img1 = sha.enhance(randsha)
        return sha_img1
    elif rand_choice == 1:
        out1 = img.rotate(rand)
        bri = ImageEnhance.Brightness(out1)
        bri_img1 = bri.enhance(randbri)
        col = ImageEnhance.Color(bri_img1)
        col_img1 = col.enhance(randcol)
        con = ImageEnhance.Contrast(col_img1)
        con_img1 = con.enhance(randcon)
        sha = ImageEnhance.Sharpness(con_img1)
        sha_img1 = sha.enhance(randsha)
        return sha_img1

def clc_total_correct_predict(predicts, labels):
    """
    统计模型所有标签全部预测正确的个数
    :param predicts: 预测的标签[N.C]
    :param labels: 正确的标签[N.C]
    :return: 标签全部预测正确的个数
    """
    total_correct = 0

    dim0 = labels.size(0)
    for i in range(dim0):
        if torch.equal(predicts[i], labels[i]):
            total_correct += 1.0
        else:
            pass
    return total_correct
def predict_transform(outputs, confidence):
    """
    将模型的输出经过阈值筛选变化为0或者1
    :param outputs: 模型的输出值[N,C]
    :param confidence: 阈值，大于该阈值的值为1,小于为0
    :return: 最终的预测值[N,C]
    """
    # gt,将大于confidence变为True,否则为false,然后经过float()变为1和0
    predicts = outputs.gt(confidence).float()

    return predicts
class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr=0.6, p=0.9):  # snr越大，增强效果越弱
        assert isinstance(snr, float) and (isinstance(p, float))  # 2020 07 26 or --> and
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img
## paste box img to balnk-img
# def paste2blank(img,boxs=None,img_blank=None): #box (y1,x2,y2,x1)
#     img = cv2.imread(img)
#     size = img.shape
#     img_blank = cv2.imread(img_blank)
#     img_blank = img_blank.copy()
#     img_blank = cv2.resize(img_blank,(size[1],size[0]))
#     for i in boxs:
#         y1=i[0]
#         x2=i[1]
#         y2=i[2]
#         x1=i[3]
#         img_blank[y1:y2,x1:x2]=img[y1:y2,x1:x2]
#     cv2.imshow('blank',img_blank)
#     cv2.waitKey(0)
#     return img_blank
#
# ## 将多个矩形框区域bouding-box，在图片上凸显
# def hilight_img_box(img='1.png',boxs=[(100,300,200,0)],outpath=None): #box (y1,x2,y2,x1) ## y1,x1,h,w
#     img = cv2.imread(img)
#     size = img.shape[0:2]
#     ones_img = numpy2cv(size)
#     assert ones_img.shape==img.shape
#     # img_copy = img.copy()
#
#     for i in boxs:
#         y1=i[0]
#         x1=i[1]
#         y2=y1+i[2]
#         x2=x1+i[3]
#         ones_img[y1:y2,x1:x2]=1.7
#     img = img*ones_img
#     if outpath:
#         cv2.imwrite(outpath,img)
#     del ones_img
#     return img

# hilight_img_box()



def decode_yuv(yuv_path, width=240, height=320):
    yuv_frame = np.fromfile(yuv_path, dtype=np.uint8)
    # print(yuv_path)
    # print(width)
    # print(height)
    # print('*****')
    yuv_frame = yuv_frame.reshape((height, width))
    # cv2.imwrite("yuv.jpg",yuv_frame)
    img_pil = Image.fromarray(yuv_frame).convert('RGB')
    # img_pil.save(save_path)
    # image_arr = np.array(image, dtype='uint8')
    return img_pil

    # np.save(save_path + '.npy', image_arr)




if __name__ == "__main__":
    yuv_path = "testdata/32_965_27_50.yuv"
    # save_path = "./save/"
    decode_yuv(yuv_path, save_path='x.png',width=27,height=50)






