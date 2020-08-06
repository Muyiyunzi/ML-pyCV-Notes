# -*- coding:utf-8 -*-
from PIL import Image, ImageDraw
import imtools
from pylab import *
import pca

# 获取selected-fontimages文件下的图像文件名，并保存在list中
imlist = imtools.get_imlist('a_selected_thumbs')
imnbr = len(imlist)

# 创建矩阵，存储所有拉成一组形式后的图像
immatrix = array([array(Image.open(im)).flatten()
                  for im in imlist],'f')

V, S, immean = pca.pca(immatrix)

projected = array([dot(V[[1,2]],immatrix[i]-immean) for i in range(imnbr)])

# 高和宽
h,w = 1200,1200

# 创建一幅白色背景图
img = Image.new('RGB',(w,h),(255,255,255))
draw = ImageDraw.Draw(img)

# 绘制坐标轴
draw.line((0, h/2, w, h/2), fill=(255, 0, 0))
draw.line((w/2, 0, w/2, h), fill=(255, 0, 0))

# 缩放以适应坐标系
scale = abs(projected).max(0)
scaled = floor(array([ (p/scale) * (w/2 - 20, h/2 - 20) + (w/2, h/2) for p in projected])).astype(int)

# 粘贴每幅图像的缩略图到白色背景图片
for i in range(imnbr):
    nodeim = Image.open(imlist[i])
    nodeim.thumbnail((25,25))
    ns = nodeim.size
    box = (scaled[i][0] - ns[0] // 2, scaled[i][1] - ns[1] // 2, scaled[i][0] + ns[0] // 2 + 1, scaled[i][1] + ns[1] // 2 + 1)
    img.paste(nodeim, box)

img.save('pca_font.jpg')

figure()
imshow(img)
axis('off')
show()