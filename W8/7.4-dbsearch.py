# -*- coding:utf-8 -*-

import imagesearch
import sift
import imtools
import vocabulary
import pickle


imlist = imtools.get_imlist('first500')

nbr_images = len(imlist)
featlist = [ imlist[i][:-3]+'sift' for i in range(nbr_images)]
'''
for i in range(nbr_images):
    sift.process_image(imlist[i],featlist[i])
'''
f = open('vocabulary.pkl', 'rb')
voc = pickle.load(f)
f.close()

src = imagesearch.Searcher('test.db', voc)
locs, descr = sift.read_features_from_file(featlist[0])
iw = voc.project(descr)
print 'ask using a histogram...'
print src.candidates_from_histogram(iw)[:10]

print '\n==================\n'

print 'try a query...'
print src.query(imlist[0])[:10]

print 'score = ', imagesearch.compute_ukbench_score(src, imlist[:4])

nbr_results = 6
res = [w[1] for w in src.query(imlist[0])[:nbr_results]]
imagesearch.plot_results(src,res)