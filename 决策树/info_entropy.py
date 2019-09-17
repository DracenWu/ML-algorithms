import numpy as np
from collections import Counter


# 按照标签分类 计算信息熵
def info_entropy(dataset: 'numpy.ndarray'):
    num = dataset.shape[0]  # 样本数
    counter = Counter(dataset[:, -1])  # 统计最后一列的标签
    distribution = np.array(list(counter.values()))  # 获得标签分类数量
    p = distribution / num  # 获得标签的概率分布
    entropy = -(p * np.log2(p)).sum()  # 求信息熵
    return entropy

#返回最大信息增益的列标
def best_info_gains(dataset):
    baseEnt = info_entropy(dataset)  # 根节点信息熵
    ents = [0 for _ in range(dataset.shape[1] - 1)]  # 存放特征信息熵

    for i in range(dataset.shape[1] - 1):  # 对数据列循环
        labels = dataset[:, i]  # 获取列数据
        label_set = set(labels)  # 获取列分类
        for label in label_set:  # 对列分类循环
            subsets = dataset[dataset[:, i] == label]  # 获取满足当前分类的数据
            ent = info_entropy(subsets)  # 计算当前分类的信息熵
            ents[i] += (subsets.shape[0] / dataset.shape[0]) * ent  # 计算当前特征的信息熵
    gains = baseEnt - ents  # 计算信息增益
    return np.argmax(gains)  # 返回最大信息增益的列标
