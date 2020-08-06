# -*- coding:utf-8 -*-

"""
Function: figure 6.4
    Clustering of pixels based on their color value using k-means.
"""
from scipy.cluster.vq import *
from scipy.misc import imresize
from pylab import *
from PIL import Image

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

def clusterpixels(infile, k, steps):
    im = array(Image.open(infile))
    dx = im.shape[0] / steps
    dy = im.shape[1] / steps
    # compute color features for each region
    features = []
    for x in range(steps):
        for y in range(steps):
            R = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
            G = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
            B = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
            features.append([R, G, B])
    features = array(features, 'f')     # make into array
    # 聚类， k是聚类数目
    centroids, variance = kmeans(features, k)
    code, distance = vq(features, centroids)
    # create image with cluster labels
    codeim = code.reshape(steps, steps)
    codeim = imresize(codeim, im.shape[:2], 'nearest')
    return codeim

k=3
infile_empire = 'empire.jpg'
im_empire = array(Image.open(infile_empire))

infile_boy_on_hill = 'boy_on_hill.jpg'
im_boy_on_hill = array(Image.open(infile_boy_on_hill))

steps = (50, 100)  # image is divided in steps*steps region
print steps[0], steps[-1]

#显示原图empire.jpg
figure()
subplot(231)
title(u'原图', fontproperties=font)
axis('off')
imshow(im_empire)

# 用50*50/100*100的块对empire.jpg的像素进行聚类
for i in range(2):
    codeim= clusterpixels(infile_empire, k, steps[i])
    subplot(232+i)
    title(u'k=3,steps=%d' % (50*(i+1)), fontproperties=font)
    axis('off')
    imshow(codeim)


#显示原图boyonhill.jpg
subplot(234)
title(u'原图', fontproperties=font)
axis('off')
imshow(im_boy_on_hill)

# 用50*50/100*100的块对boyonhill.jpg的像素进行聚类
for i in range(2):
    codeim= clusterpixels(infile_boy_on_hill, k, steps[i])
    subplot(235+i)
    title(u'k=3,steps=%d' % (50*(i+1)), fontproperties=font)
    axis('off')
    imshow(codeim)


show()