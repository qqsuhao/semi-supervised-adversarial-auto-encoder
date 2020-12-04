# -*- coding:utf8 -*-
# @TIME     : 2020/11/1 14:00
# @Author   : Hao Su
# @File     : model_test.py

import numpy as np
from visualization.visual import plot_tsne, plot_2D_scatter
from sklearn.metrics import classification_report


class global_index():
    '''
    用于存放全局指标
    '''

    def __init__(self):
        self.predictions = []
        self.targets = []
        self.latent = []


    def add_real_targets(self, real):
        '''
        :param fake: 预测标签
        :param real: 真实标签
        :return:
        '''
        if isinstance(real, list):
            self.targets += real


    def add_fake_targets(self, fake):
        if isinstance(fake, list):
            self.predictions += fake


    def add_letent_z(self, z):
        self.latent.append(z)


    def classification_accuracy(self):
        '''
        :param enc: model
        :return: accuracy
        '''
        cr = classification_report(self.targets, self.predictions, output_dict=True)
        return cr["accuracy"]


    def plot_latent_distribution(self, title, path):
        Z = np.concatenate([i for i in self.latent], axis=0)
        if not Z.shape[0] == len(self.targets):
            print("number of latent z doesnot match number of labels")
            return
        plot_tsne(Z, self.targets, title, path)


    def plot_2D_scatter(self, title, path):
        Z = np.concatenate([i for i in self.latent], axis=0)
        if Z.shape[1] == 2:
            if not Z.shape[0] == len(self.targets):
                print("number of latent z doesnot match number of labels")
                return
            plot_2D_scatter(Z, self.targets, title, path)