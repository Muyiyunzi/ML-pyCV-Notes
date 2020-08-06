# -*- coding:utf-8 -*-

import pickle
import sift
import imagesearch
import homography
import imtools

# 载入图像列表和词汇
imlist = imtools.get_imlist('first500')

nbr_images = len(imlist)
featlist = [ imlist[i][:-3]+'sift' for i in range(nbr_images)]

with open('vocabulary.pkl', 'rb') as f:
    voc = pickle.load(f)

src = imagesearch.Searcher('test.db', voc)

# 查询图像的索引号和返回的搜索结果数目
q_ind = 50
nbr_results = 20

# 常规查询
res_reg = [w[1] for w in src.query(imlist[q_ind])[:nbr_results]]
print 'top matches (regular):', res_reg

# 载入查询图像特征
q_locs,q_descr = sift.read_features_from_file(featlist[q_ind])
fp = homography.make_homog(q_locs[:,:2].T)

# 用RANSAC 模型拟合单应性
model = homography.RansacModel()

rank = {}
# 载入搜索结果的图像特征
for ndx in res_reg[1:]:
    locs,descr = sift.read_features_from_file(featlist[ndx])

    # 获取匹配数
    matches = sift.match(q_descr,descr)
    ind = matches.nonzero()[0]
    ind2 = matches[ind]
    tp = homography.make_homog(locs[:,:2].T)

    # 计算单应性，对内点计数。如果没有足够的匹配数则返回空列表
    try:
        H,inliers = homography.H_from_ransac(fp[:,ind],tp[:,ind2],model,match_theshold=4)
    except:
        inliers = []

    # 存储内点数
    rank[ndx] = len(inliers)

# 将字典排序，以首先获取最内层的内点数
sorted_rank = sorted(rank.items(), key=lambda t: t[1], reverse=True)
res_geom = [res_reg[0]] + [s[0] for s in sorted_rank]
print 'top matches (homography):', res_geom

# 显示靠前的搜索结果
for i in range(8):
    imagesearch.plot_results(src, res_reg[i:i+1])
    imagesearch.plot_results(src, res_geom[i:i+1])
