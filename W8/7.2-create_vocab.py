# -*- coding:utf-8 -*-
from PIL import Image
import imtools
import pickle
import vocabulary
import sift

# 获取selected-fontimages文件下的图像文件名，并保存在list中
imlist = imtools.get_imlist('first500')

nbr_images = len(imlist)
featlist = [ imlist[i][:-3]+'sift' for i in range(nbr_images)]

for i in range(nbr_images):
    sift.process_image(imlist[i],featlist[i])

voc = vocabulary.Vocabulary('ukbenchtest')
voc.train(featlist, 500, 10) #书中为1000，这里只取500个应该改成500吧

# 保存词汇
with open('vocabulary.pkl', 'wb') as f:
    pickle.dump(voc, f)

print 'vocabulary is:', voc.name, voc.nbr_words

