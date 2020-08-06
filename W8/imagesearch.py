# -*- coding:utf-8 -*-

import pickle
from pylab import *
from sqlite3 import dbapi2 as sqlite
from PIL import Image

class Indexer(object):
    def __init__(self, db, voc):
        """ 初始化数据库的名称及词汇对象 """
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def get_id(self, imname):
        """ 获取图像id，如果不存在，就进行添加"""

        cur = self.con.execute(
        "select rowid from imlist where filename='%s'" % imname)
        res = cur.fetchone()
        if res == None:
            cur = self.con.execute(
            "insert into imlist(filename) values ('%s')" % imname)
            return cur.lastrowid

        else:
            return res[0]

    def is_indexed(self, imname):
        """ 如果图像名字（imname）被索引到，就返回True"""

        im = self.con.execute("select rowid from imlist where filename = '%s'" % imname).fetchone()
        return im != None

    def db_commit(self):
        self.con.commit()

    def create_tables(self):
        """ 创建数据库表单"""
        self.con.execute('create table imlist(filename)')
        self.con.execute('create table imwords(imid,wordid,vocname)')
        self.con.execute('create table imhistograms(imid,histogram,vocname)')
        self.con.execute('create index im_idx on imlist(filename)')
        self.con.execute('create index wordid_idx on imwords(wordid)')
        self.con.execute('create index imid_idx on imwords(imid)')
        self.con.execute('create index imidhist_idx on imhistograms(imid)')
        self.db_commit()

    def add_to_index(self,imname,descr):
        """ 获取一幅带有特征描述子的图像，投影到词汇上并添加进数据库"""
        if self.is_indexed(imname): return
        print 'indexing', imname

        # 获取图像id
        imid = self.get_id(imname)

        # 获取单词
        imwords = self.voc.project(descr)
        nbr_words = imwords.shape[0]

        # 将每个单词与图像链接起来
        for i in range(nbr_words):
            word = imwords[i]
            # wordid 就是单词本身的数字
            self.con.execute("insert into imwords(imid, wordid, vocname) values(?, ?, ?)", (imid,word,self.voc.name))

        # 存储图像的单词直方图
        # 用pickle 模块将NumPy 数组编码成字符串
        self.con.execute("insert into imhistograms(imid,histogram,vocname) values(?, ?, ?)", (imid,pickle.dumps(imwords), self.voc.name))

class Searcher(object):
    def __init__(self, db, voc):
        """ 初始化数据库的名称 """
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def candidates_from_word(self, imword):
        """ G 获取包含imword 的图像列表"""

        im_ids = self.con.execute(
            "select distinct imid from imwords where wordid=%d" % imword).fetchall()
        return [i[0] for i in im_ids]

    def candidates_from_histogram(self, imwords):
        """ 获取具有相似单词的图像列表"""
        # 获取单词id
        words = imwords.nonzero()[0]

        # 寻找候选图像
        candidates = []
        for word in words:
            c = self.candidates_from_word(word)
            candidates+=c

        # 获取所有唯一的单词，并按出现次数反向排序
        tmp = [(w,candidates.count(w)) for w in set(candidates)]
        tmp.sort(cmp=lambda x, y:cmp(x[1],y[1]))
        tmp.reverse()

        # 返回排序后的列表，最匹配的排在最前面
        return [w[0] for w in tmp]

    def get_imhistogram(self, imname):
        """ 返回一幅图像的单词直方图"""

        im_id = self.con.execute(
            "select rowid from imlist where filename='%s'" % imname).fetchone()
        s = self.con.execute(
            "select histogram from imhistograms where rowid='%d'" % im_id).fetchone()

        # 用pickle 模块从字符串解码Numpy 数组
        return pickle.loads(str(s[0]))

    def query(self, imname):
        """ 查找所有与imname 匹配的图像列表"""

        h = self.get_imhistogram(imname)
        candidates = self.candidates_from_histogram(h)

        matchscores = []

        for imid in candidates:
            # 获取名字
            cand_name = self.con.execute(
                "select filename from imlist where rowid=%d" % imid).fetchone()
            cand_h = self.get_imhistogram(cand_name)
            cand_dist = sqrt(sum(self.voc.idf * (h - cand_h)**2 ) )  # 用L2 距离度量相似性
            matchscores.append((cand_dist, imid))

        # 返回排序后的距离及对应数据库ids 列表
        matchscores.sort()
        return matchscores

    def get_filename(self, imid):
        """ 返回图像id 对应的文件名"""
        s = self.con.execute(
            "select filename from imlist where rowid='%d'" % imid).fetchone()
        return s[0]

def compute_ukbench_score(src, imlist):
    """ 对查询返回的前4 个结果计算平均相似图像数，并返回结果"""

    nbr_images = len(imlist)
    pos = zeros((nbr_images, 4))

    # 获取每幅查询图像的前4 个结果
    for i in range(nbr_images):
        pos[i] = [w[1] - 1 for w in src.query(imlist[i])[:4]]

    # 计算分数，并返回平均分数
    score = array([(pos[i] // 4) == (i // 4) for i in range(nbr_images)]) * 1.0
    return sum(score) / (nbr_images)

def plot_results(src,res):
    """ 显示在列表res 中的图像"""
    figure()
    nbr_results = len(res)
    for i in range(nbr_results):
        imname = src.get_filename(res[i])
        subplot(1,nbr_results,i+1)
        imshow(array(Image.open(imname)))
        axis('off')
    show()

