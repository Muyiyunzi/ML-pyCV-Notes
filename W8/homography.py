# -*- coding: utf-8 -*-
from numpy import *
from PIL import Image
from pylab import *

def normalize(points):
    """ 在齐次坐标意义下，对点集进行归一化，使最后一行为1 """
    for row in points:
        row /= points[-1]
    return points

def make_homog(points):
    """ 将点集（dim×n 的数组）转换为齐次坐标表示 """
    return vstack((points,ones((1,points.shape[1]))))

def H_from_points(fp,tp):
    """ 使用线性DLT 方法，计算单应性矩阵H，使fp 映射到tp。点自动进行归一化"""
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # 对点进行归一化（对数值计算很重要）
    # --- 映射起始点---
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = dot(C1,fp)

    # --- 映射对应点---
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1 / maxstd, 1 / maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp = dot(C2, tp)

    # 创建用于线性方法的矩阵，对于每个对应对，在矩阵中会出现两行数值
    nbr_correspondences = fp.shape[1]
    A = zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0,
                    tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
                    tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]

    U, S, V = linalg.svd(A)
    H = V[8].reshape((3, 3))

    # 反归一化
    H = dot(linalg.inv(C2), dot(H, C1))

    # 归一化，然后返回
    return H / H[2, 2]

def Haffine_from_points(fp,tp):
    """ 计算H，仿射变换，使得tp 是fp 经过仿射变换H 得到的"""
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # 对点进行归一化
    # --- 映射起始点---
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = dot(C1,fp)

    # --- 映射对应点---
    m = mean(tp[:2], axis=1)
    C2 = C1.copy() # 两个点集，必须都进行相同的缩放
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = dot(C2,tp)

    # 因为归一化后点的均值为0，所以平移量为0
    A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = linalg.svd(A.T)

    # 创建矩阵B 和C
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1)
    H = vstack((tmp2,[0,0,1]))

    # 反归一化
    H = dot(linalg.inv(C2),dot(H,C1))

    return H / H[2,2]

class RansacModel(object):
    """ 用于测试单应性矩阵的类，其中单应性矩阵是由网站http://www.scipy.org/Cookbook/RANSAC上的ransac.py 计算出来的"""
    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """ 计算选取的4个对应的单应性矩阵 """
        # 将其转置，来调用H_from_points() 计算单应性矩阵
        data = data.T
        # 映射的起始点
        fp = data[:3,:4]
        # 映射的目标点
        tp = data[3:,:4]
        # 计算单应性矩阵，然后返回
        return H_from_points(fp,tp)

    def get_error( self, data, H):
        """ 对所有的对应计算单应性矩阵，然后对每个变换后的点，返回相应的误差"""
        data = data.T

        # 映射的起始点
        fp = data[:3]
        # 映射的目标点
        tp = data[3:]

        # 变换fp
        fp_transformed = dot(H,fp)

        # 归一化齐次坐标
        for i in range(3):
            fp_transformed[i] /= fp_transformed[2]

        # 返回每个点的误差
        return sqrt( sum((tp-fp_transformed)**2,axis=0) )

def H_from_ransac(fp,tp,model,maxiter=1000,match_theshold=10):
    """ 使用RANSAC 稳健性估计点对应间的单应性矩阵H（ransac.py 为从
    http://www.scipy.org/Cookbook/RANSAC 下载的版本）"""

    # 输入：齐次坐标表示的点fp，tp（3×n 的数组）
    import ransac

    # 对应点组
    data = vstack((fp,tp))
    # 计算H，并返回
    H, ransac_data = ransac.ransac(data.T, model, 4, maxiter, match_theshold, 10,
                                  return_all=True)
    return H #, ransac_data['inliers']