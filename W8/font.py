# -*- coding:utf-8 -*-

from PIL import Image
from numpy import *
from pylab import *
import pca
import imtools
import pickle

imlist = imtools.get_imlist('fontimages/a_thumbs') #记得加引号才能表path

im = array(Image.open(imlist[0])) #打开一幅图像，获取其大小
m,n = im.shape[0:2] # 获取图像的大小
imnbr = len(imlist) # 获取图像的数目

immatrix = array([array(Image.open(im)).flatten()
                  for im in imlist], 'f')

# 执行PCA操作
V, S, immean = pca.pca(immatrix)

# 显示一些图像（均值图像和前7个模式）
figure()
gray()
subplot(2, 4, 1)
imshow(immean.reshape(m, n))
for i in range(7):
    subplot(2, 4, i + 2)
    imshow(V[i].reshape(m, n))

show()

f = open('a_pca_modes.pkl', 'wb')
pickle.dump(immean, f)
pickle.dump(V, f)
f.close()


