# -*- coding:utf-8 -*-

from PIL import Image
from numpy import *

def pca(X):
    """ 主成分分析：
    输入：矩阵X ，其中该矩阵中存储训练数据，每一行为一条训练数据
    返回：投影矩阵（按照维度的重要性排序）、方差和均值"""

    # 获取维数
    num_data, dim = X.shape

    # 数据中心化
    mean_X = X.mean(axis = 0)
    X = X - mean_X

    if dim > num_data:
        # PCA-使用紧致技巧
        M = dot(X, X.T) # 协方差矩阵
        e, EV = linalg.eigh(M) # 特征值和特征向量
        tmp = dot(X.T, EV).T
        V = tmp[::-1] # 将最后的特征向量逆转
        S = sqrt(e)[::-1] # 特征值是按递增顺序排序的，需要逆转
        for i in range(V.shape[1]):
            V[:,i] /= S

    else:
        # PCA-使用SVD方法
        U, S, V = linalg.svd(X)
        V = V[:num_data] # 仅仅返回前num_data维的数据才合理

    # 返回投影矩阵、方差和均值
    return V, S, mean_X


def center(X):
    """    Center the square matrix X (subtract col and row means). """

    n, m = X.shape
    if n != m:
        raise Exception('Matrix is not square.')

    colsum = X.sum(axis=0) / n
    rowsum = X.sum(axis=1) / n
    totalsum = X.sum() / (n ** 2)

    # center
    Y = array([[X[i, j] - rowsum[i] - colsum[j] + totalsum for i in range(n)] for j in range(n)])

    return Y

