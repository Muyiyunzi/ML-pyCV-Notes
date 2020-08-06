# -*- coding:utf-8 -*-
import pickle
import sift
import imagesearch
from sqlite3 import dbapi2 as sqlite
import imtools

# 获取图像列表
imlist = imtools.get_imlist('first500')
nbr_images = len(imlist)

# 获取特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

# 载入词汇
with open('vocabulary.pkl', 'rb') as f:
    voc = pickle.load(f)

# 创建索引器
indx = imagesearch.Indexer('test.db', voc)
indx.create_tables()

# 遍历整个图像库，将特征投影到词汇上并添加到索引中
for i in range(nbr_images)[:1000]:
    locs, descr = sift.read_features_from_file(featlist[i])
    indx.add_to_index(imlist[i],descr)

# 提交到数据库
indx.db_commit()
con = sqlite.connect('test.db')
print con.execute('select count (filename) from imlist').fetchone()
print con.execute('select * from imlist').fetchone()