# -*- coding:utf-8 -*-

import imtools
from PIL import Image
from scipy.cluster.vq import *
from pylab import *
import pickle

# 获取selected-fontimages文件下的图像文件名，并保存在list中
imlist = imtools.get_imlist('a_selected_thumbs')
imnbr = len(imlist)

# 载入模型文件
with open("a_pca_modes.pkl", 'rb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)

# 创建矩阵，存储所有拉成一组形式后的图像
immatrix = array([array(Image.open(im)).flatten()
                  for im in imlist],'f')

# 投影到前40个主成分上
immean = immean.flatten()
projected = array([dot(V[:40], immatrix[i]-immean) for i in range(imnbr)])

# 进行k-means聚类
projected = whiten(projected)
centroids, distortion = kmeans(projected, 4)

code, distance = vq(projected, centroids)

# 绘制聚类簇
for k in range(4):
    ind = where(code == k)[0]
    figure()
    gray()
    for i in range(minimum(len(ind), 40)):
        subplot(4, 10, i+1)
        imshow(immatrix[ind[i]].reshape((25, 25)))
        axis('off')

show()

