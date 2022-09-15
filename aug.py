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
from efficientnet_pytorch import EfficientNet
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



