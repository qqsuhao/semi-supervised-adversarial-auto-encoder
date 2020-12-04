# -*- coding:utf8 -*-
# @TIME     : 2020/11/6 15:18
# @Author   : Hao Su
# @File     : self_transforms.py

'''
自定义的transform
'''


import numpy as np
from PIL import Image
import cv2

class AddSaltPepperNoise(object):
    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)
        c, h, w = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[Nd/2.0, Nd/2.0, Sd])      # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)
        img[mask == 0] = 0                                                              # 椒
        img[mask == 1] = 1
        return img


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        c, h, w = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(1, h, w))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 1] = 1                       # 避免有值超过255而反转
        img = img.astype('float32')
        return img


class Fllip(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        img = 1 - img
        return img


class Gammma(object):
    def __init__(self, gamma):
        self.gamma = gamma


    def __call__(self, img):
        img = np.array(img).astype("uint8")
        # img_gamma = np.power(img, self.gamma)
        # print(img_gamma)
        img_gamma = cv2.equalizeHist(img)
        return img_gamma.astype('float32')
