# -*- coding:utf8 -*-
# @TIME     : 2020/10/31 17:19
# @Author   : Hao Su
# @File     : visual.py


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_embedding(data, label, title, path):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    fig, ax = plt.subplots(dpi=300)
    scatter = ax.scatter(data[:, 0], data[:, 1], marker='.', c=label, cmap='tab10')
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(path)
    plt.close()
    # return fig


def plot_tsne(data, label, title, path):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    plot_embedding(result, label, title, path)


def plot_2D_scatter(data, label, title, path):
    if len(data.shape) != 2:
        return
    if data.shape[1] > 2:
        return
    fig, ax = plt.subplots(dpi=300)
    scatter = ax.scatter(data[:, 0], data[:, 1], marker='.', c=label, cmap='tab10')
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(path)
    plt.close()


