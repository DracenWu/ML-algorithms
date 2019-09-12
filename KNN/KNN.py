import numpy as np
from collections import Counter


# 封装成类
class KNN(object):
    def __init__(self, targets, sources, k):
        self.targets = targets  # 目标数据
        self.sources = sources  # 源数据
        self.k = k  # 取前k个距离的特征点

    # 计算欧氏距离
    # 利用numpy矩阵的特点直接进行矩阵加减
    def __distance(self, feats, target):
        return list((((feats - target) ** 2).sum(1)) ** 0.5)

    def __knn(self, target, source, k):
        sourcefeats = source[:, 1:-1].astype(np.int)  # 获取源数据特征
        sourcelabels = list(source[:, -1])  # 获取源数据标签
        dists = self.__distance(sourcefeats, target)  # 计算欧氏距离
        dists = {k: v for k, v in zip(dists, sourcelabels)}  # 打包距离和标签
        dists = {k: dists[k] for k in sorted(dists.keys())}  # 根据距离排序
        results = [k for k, v in Counter(list(dists.values())[:k]).items()]  # 统计前k个标签的数量，按降序排列
        return results[0]  # 返回概率最大的标签

    def knn(self):
        for t in self.targets:
            label = self.__knn(t[1:-1], self.sources, self.k)  # 获得最大概率标签
            t[-1] = label  # 赋值
        print('result\n', self.targets)


# 单独的函数
def _knn(target, source, k):
    feats = source[:, 1:-1].astype(np.int)  # 获取源数据特征
    labels = list(source[:, -1])  # 获取源数据标签
    dists = list((((feats - target) ** 2).sum(1)) ** 0.5)  # 计算欧氏距离
    dists = {k: v for k, v in zip(dists, labels)}  # 打包距离和标签
    dists = {k: dists[k] for k in sorted(dists.keys())}  # 根据距离排序
    results = [k for k, v in Counter(list(dists.values())[:k]).items()]  # 统计前k个标签的数量，按降序排列
    return results[0]  # 返回概率最大的标签


# 单独的函数
def knn(targets, sources, k):
    for t in targets:
        label = _knn(t[1:-1], sources, k)
        t[-1] = label
    print(targets)

#构造数据
def createSource(source):
    return np.array(source).T


if __name__ == '__main__':
    data = [
        ['无问西东', '后来的我们', '前任3', '红海行动', '唐人街探案', '战狼2'],
        [1, 5, 12, 108, 112, 115],
        [101, 89, 97, 5, 9, 8],
        [2, 3, 20, 68, 89, 100],
        ['爱情片', '爱情片', '爱情片', '动作片', '动作片', '动作片'],
    ]
    sources = createSource(data)
    print('source data\n', sources)
    targets = np.array([['喜洋洋', 24, 60, 20, None],
                        ['美羊羊', 54, 1, 30, None],
                        ['懒羊羊', 32, 20, 40, None],
                        ['沸羊羊', 94, 60, 10, None]])
    print('target data\n', targets)

    k = KNN(targets, sources, 3)
    k.knn()
