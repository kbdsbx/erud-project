import cv2
import os
import numpy as np
import json
import math
import time

def getX(count = 4521) :
    """
    获取图片样本
    
    count, int 图片样本数
    """
    path = "D:\Study\proj\static\proj-caches"


    x = np.zeros((count, 128, 192, 3))
    for i in range(count) :
        x[i] = cv2.transpose(cv2.imread(os.path.join(path, str(i+1) + '.jpg'))).tolist()
    
    return np.array(x)

def getY(count = 4521) :
    """
    获取图片标签
    
    count, int 图片样本数
    """
    labelfile = "D:\Study\proj\static\labelfile.json"
    with open (labelfile, 'r', encoding='utf-8') as f:
        labels = json.loads(f.read())
    
    return np.array(labels[:count]).reshape((count, 1))


def process_samples(X, Y, s, split=(0.8, 0.1, 0.1)) :
    """
    样本切分成多个小集合

    * X 样本集
    * Y 标签集
    * s 分量

    返回值
    * batches (tX, tY) batchs
    * test_X
    * test_Y one-hot向量组
    * verify_X
    * verify_Y one-hot向量组
    """

    m = X.shape[0]
    # one_hot_Y = np.eye(5)[Y.reshape(-1)]

    np.random.seed(1)

    permutation = list(np.random.permutation(m))
    # permutation = [i for i in range(m)]
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation]

    sp1 = math.floor(split[0] * m)
    sp2 = math.floor((split[0] + split[1]) * m)

    # 将数据集合分成训练集、测试集、验证集
    # 训练集
    train_X = shuffled_X[0:sp1, :, :, :]
    train_Y = shuffled_Y[0:sp1]
    train_Y_one_hot = np.eye(5)[train_Y.reshape(-1)]
    train_idx = permutation[0:sp1]
    # 测试集
    test_X = shuffled_X[sp1:sp2, :, :, :]
    test_Y = shuffled_Y[sp1:sp2]
    test_idx = permutation[sp1:sp2]
    # 验证集
    verify_X = shuffled_X[sp2:, :, :, :]
    verify_Y = shuffled_Y[sp2:]
    verify_idx = permutation[sp2:]

    # 将训练集分成多个bench
    train_m = train_X.shape[0]
    n = math.floor(train_m/s)
    batches = []

    for k in range(0, n) :
        mX = train_X[k*s : (k+1) * s, :, :, :]
        mY = train_Y_one_hot[k*s : (k+1) * s, :]

        miniB = (mX, mY)
        batches.append(miniB)
    
    if train_m % s != 0 :
        mX = train_X[train_m - (train_m % s):, :, :, :]
        mY = train_Y_one_hot[train_m - (train_m % s):, :]

        miniB = (mX, mY)
        batches.append(miniB)

    
    return batches, train_X, train_Y, test_X, test_Y, verify_X, verify_Y, train_idx, test_idx, verify_idx

# X = getX(10)
# Y = getY(10)

# batches, train_X, train_Y, test_X, test_Y, verify_X, verify_Y =  process_samples(X, Y, 2)
# print(len(batches))
# print(train_X.shape)
# print(train_Y.shape)
# print(test_X.shape)
# print(test_Y.shape)
# print(verify_X.shape)
# print(verify_Y.shape)


import sys
sys.path.append('D:\Study\erud')
import erud


# conv(128 * 192 * 3) x 2 | max_pool_v3 x 1 | batchnorm
# conv(64 * 96 * 8) x 2 | max_pool_v3 x 1 | batchnorm
# conv(32 * 48 * 16) x 2 | max_pool_v3 x 1 | batchnorm
# conv(16 * 24 * 32) x 2 | max_pool_v3 x 1 | batchnorm
# conv(8 * 12 * 64) x 2 | max_pool_v3 x 1 | batchnorm
# conv(4 * 6 * 128) x 2 | max_pool_v3 x 1 | batchnorm
# Y: (无，黑丝，白丝，花丝，裸足/肉丝)

# VGG-16 的魔改版
# 294552 + 15360 + 5 = 309917 个参数
def stocking_classification () :
   
    path = __file__[:__file__.rfind('\\')]

    # 加载所有样本
    print("Samples loading...")
    X = getX(4521)
    Y = getY(4521)
    batches, train_X, train_Y, test_X, test_Y, verify_X, verify_Y, train_idx, test_idx, verify_idx =  process_samples(X, Y, 64)
    print("%d samples loaded." %(X.shape[0]))

    gtest = erud.nous(
        '''
        X ->

            conv2d_v3_same(1) W11 -> conv2d_v3_same(1) W12 -> max_pool_v3(2, 2, 2) -> batchnorm ->

            conv2d_v3_same(1) W21 -> conv2d_v3_same(1) W22 -> max_pool_v3(2, 2, 2) -> batchnorm ->

            conv2d_v3_same(1) W31 -> conv2d_v3_same(1) W32 -> max_pool_v3(2, 2, 2) -> batchnorm ->

            conv2d_v3_same(1) W41 -> conv2d_v3_same(1) W42 -> max_pool_v3(2, 2, 2) -> batchnorm ->

            conv2d_v3_same(1) W51 -> conv2d_v3_same(1) W52 -> max_pool_v3(2, 2, 2) -> batchnorm ->

            flatten -> matmul W6 add b6 ->

        max_index(1) -> J:$$
        '''
    ).parse()

    gtest.setData('X', test_X)
    # gtest.setData('Y', test_Y)

    n, obj = erud.nous_imports(path + '/cache.json')
    g = n.g
    gtest.setData('W11', g.getData('W11'))
    gtest.setData('W12', g.getData('W12'))
    gtest.setData('W21', g.getData('W21'))
    gtest.setData('W22', g.getData('W22'))
    gtest.setData('W31', g.getData('W31'))
    gtest.setData('W32', g.getData('W32'))
    gtest.setData('W41', g.getData('W41'))
    gtest.setData('W42', g.getData('W42'))
    gtest.setData('W51', g.getData('W51'))
    gtest.setData('W52', g.getData('W52'))
    gtest.setData('W6', g.getData('W6'))
    gtest.setData('b6', g.getData('b6'))
    gtest.fprop()
    print('List of label:')
    yhats = gtest.getData('J')
    train = test_Y
    idxs = test_idx
    enum = {
        0: "无",
        1: "黑丝",
        2: "白丝",
        3: "花丝",
        4: "裸足",
    }
    for i in range(len(yhats)) :
        if yhats[i,0] != train[i,0] :
            print("%d.jpg is not %s but %s." %(idxs[i] + 1, enum[yhats[i,0]], enum[train[i,0]]))



    # 迁移参数

    # gtest.setData('X', test_X)
    # gtest.setData('Y', test_Y)
    

    # for i in range(12) :
    #     idx = (i+1) * 10
    #     n, obj = erud.nous_imports(path + '/cache-bk-'+ str(idx) + '.json')
    #     g = n.g

    #     # 加载不同时期的学习参数
    #     gtest.setData('W11', g.getData('W11'))
    #     gtest.setData('W12', g.getData('W12'))
    #     gtest.setData('W21', g.getData('W21'))
    #     gtest.setData('W22', g.getData('W22'))
    #     gtest.setData('W31', g.getData('W31'))
    #     gtest.setData('W32', g.getData('W32'))
    #     gtest.setData('W41', g.getData('W41'))
    #     gtest.setData('W42', g.getData('W42'))
    #     gtest.setData('W51', g.getData('W51'))
    #     gtest.setData('W52', g.getData('W52'))
    #     gtest.setData('W6', g.getData('W6'))
    #     gtest.setData('b6', g.getData('b6'))

    #     # 计算不同时期的测试集精确度
    #     gtest.fprop()

    #     print('test accuracy in %d: %s' %(idx, gtest.getData('J')))


    # 分批计算训练集精度（内存不足...）

    # accs = []
    # sum_accs = 36
    # for i in range(sum_accs) :
    #     gtest.setData('X', train_X[100*i:100*(i+1), :, :, :])
    #     gtest.setData('Y', train_Y[100*i:100*(i+1)])

    #     gtest.fprop()
    #     accs.append(float(gtest.getData('J')))

    # avg_accs = sum(accs) / sum_accs
    # print('train accuracy: %s' %(avg_accs))

    # gtest.setData('X', train_X)
    # gtest.setData('Y', train_Y)

    # gtest.fprop()
    # print('train accuracy: %s' %(gtest.getData('J')))

    # 计算测试集精度
    # gtest.setData('X', test_X)
    # gtest.setData('Y', test_Y)

    # gtest.fprop()

    # print('test accuracy: %s' %(gtest.getData('J')))

    # 计算验证集精度
    # gtest.setData('X', verify_X)
    # gtest.setData('Y', verify_Y)

    # gtest.fprop()

    # print('test accuracy: %s' %(gtest.getData('J')))


stocking_classification()



