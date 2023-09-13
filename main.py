import cv2
import os
import numpy as np
import json
import math
import time

def getX(count) :
    """
    获取图片样本
    
    count, int 图片样本数
    """
    path = "D:\Study\proj\static\proj-samples-enhance"


    x = np.zeros((count, 128, 192, 3))
    for i in range(count) :
        x[i] = cv2.transpose(cv2.imread(os.path.join(path, str(i+1) + '.jpg'))).tolist()
    
    return np.array(x)

def getY(count) :
    """
    获取图片标签
    
    count, int 图片样本数
    """
    labelfile = "D:\Study\proj\static\labelfile-enhance.json"
    with open (labelfile, 'r', encoding='utf-8') as f:
        labels = json.loads(f.read())
    
    return np.array(labels[:count]).reshape((count, 1))


def process_samples(X, Y, s, split=(0.8, 0.2)) :
    """
    样本切分成多个小集合

    * X 样本集
    * Y 标签集
    * s 分量

    返回值
    * batches (tX, tY) batchs
    * test_X
    * test_Y
    """

    m = X.shape[0]
    # one_hot_Y = np.eye(5)[Y.reshape(-1)]

    np.random.seed(1)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation]

    sp1 = math.floor(split[0] * m)

    # 将数据集合分成训练集、测试集
    # 暂无验证集
    # 训练集
    train_X = shuffled_X[0:sp1, :, :, :]
    train_Y = shuffled_Y[0:sp1]
    train_Y_one_hot = np.eye(5)[train_Y.reshape(-1)]
    # 测试集
    test_X = shuffled_X[sp1:, :, :, :]
    test_Y = shuffled_Y[sp1:]

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

    
    return batches, train_X, train_Y, test_X, test_Y

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

def load_params_from_old_network(cache_file) :
    imports_str = ""
    nodes = []

    with open(cache_file, "r", encoding="utf-8") as f :
        imports_str = f.read()
    nodes = json.loads(imports_str)["nodes"]

    return nodes
    

import sys
sys.path.append('D:\Study\erud')
import erud


# conv(128 * 192 * 3) batchnorm2d x 2 | max_pool_v3 x 1 | relu
# conv(64 * 96 * 8) batchnorm2d x 2 | max_pool_v3 x 1 | relu
# conv(32 * 48 * 16) batchnorm2d x 2 | max_pool_v3 x 1 | relu
# conv(16 * 24 * 32) batchnorm2d x 2 | max_pool_v3 x 1 | relu
# conv(8 * 12 * 64) batchnorm2d x 2 | max_pool_v3 x 1 | relu
# conv(4 * 6 * 128) batchnorm2d x 2 | max_pool_v3 x 1 | relu
# flatten (3072)
# matmul(3072 * 128) + b | relu | dropout
# matmul(128 * 32) + b | relu | dropout
# matmul(32 * 5) + b | relu | dropout
# Y: (无，黑丝，白丝，花丝，裸足)

# VGG-16 的魔改版
# 294552 + 393216 + 4096 + 160 + 128 + 32 + 5 = 692189 个参数
def stocking_classification () :
    code = """
    X:(1, 128, 192, 3) ->

        conv2d_v3_same(1) W11:xavier((3, 3, 3, 8), 9):adam(0.0002) -> 
        batchnorm2d as BN11 -> 
        mul G11:randn(8):adam(0.0002) add T11:(8):adam(0.0002) -> 
        relu ->
        
        conv2d_v3_same(1) W12:xavier((3, 3, 8, 8), 9):adam(0.0002) -> 
        batchnorm2d as BN12 -> 
        mul G12:randn(8):adam(0.0002) add T12:(8):adam(0.0002) -> 
        relu ->

        max_pool_v3(2, 2, 2) ->

        

        conv2d_v3_same(1) W21:xavier((3, 3, 8, 16), 9):adam(0.0002) ->
        batchnorm2d as BN21 ->
        mul G21:randn(16):adam(0.0002) add T21:(16):adam(0.0002) ->
        relu ->

        conv2d_v3_same(1) W22:xavier((3, 3, 16, 16), 9):adam(0.0002) ->
        batchnorm2d as BN22 ->
        mul G22:randn(16):adam(0.0002) add T22:(16):adam(0.0002) ->
        relu ->

        max_pool_v3(2, 2, 2) ->

        

        conv2d_v3_same(1) W31:xavier((3, 3, 16, 32), 9):adam(0.0002) ->
        batchnorm2d as BN31 ->
        mul G31:randn(32):adam(0.0002) add T31:(32):adam(0.0002) ->
        relu ->

        conv2d_v3_same(1) W32:xavier((3, 3, 32, 32), 9):adam(0.0002) ->
        batchnorm2d as BN32 ->
        mul G32:randn(32):adam(0.0002) add T32:(32):adam(0.0002) ->
        relu ->

        max_pool_v3(2, 2, 2) ->

        

        conv2d_v3_same(1) W41:xavier((3, 3, 32, 64), 9):adam(0.0002) ->
        batchnorm2d as BN41 ->
        mul G41:randn(64):adam(0.0002) add T41:(64):adam(0.0002) ->
        relu ->

        conv2d_v3_same(1) W42:xavier((3, 3, 64, 64), 9):adam(0.0002) ->
        batchnorm2d as BN42 ->
        mul G42:randn(64):adam(0.0002) add T42:(64):adam(0.0002) ->
        relu ->

        max_pool_v3(2, 2, 2) ->

        

        conv2d_v3_same(1) W51:xavier((3, 3, 64, 128), 9):adam(0.0002) ->
        batchnorm2d as BN51 ->
        mul G51:randn(128):adam(0.0002) add T51:(128):adam(0.0002) ->
        relu ->

        conv2d_v3_same(1) W52:xavier((3, 3, 128, 128), 9):adam(0.0002) ->
        batchnorm2d as BN52 ->
        mul G52:randn(128):adam(0.0002) add T52:(128):adam(0.0002) ->
        relu ->

        max_pool_v3(2, 2, 2) ->

        

    flatten ->

        matmul W6:xavier((3072, 128), 3072) add b6:(128) -> relu ->

        matmul W7:xavier((128, 32), 128) add b7:(32) -> relu ->

        matmul W8:xavier((32, 5), 32) add b8:(5) ->

    softmax_cross_entropy(1) Y:(1, 5) -> cost -> J:$$
    """


    num_iterations = 100
    num_over_iterations = 0
    rate = 0.0002
    costs = []
    cost = 0
    m = 50
    mini = 10

    # 开启缓存
    enable_cache = True
    # 开启迁移学习
    enable_transfor = False

    # 缓存文件
    path = __file__[:__file__.rfind('\\')]
    cachefile = path + '/cache.json'

    # 迁移学习文件
    # 迁移学习文件的格式与缓存文件格式相同，可以从中读取到结构不同但参数相同的网络的参数
    transforfile = path + '/transfor.json'

    n = erud.nous()
    # 如果缓存有则从缓存加载
    if enable_cache and os.path.exists(cachefile) :
        n, obj = erud.nous_imports(cachefile)
        g = n.g
        num_over_iterations = obj['num_over_iterations']
        num_iterations = obj['num_iterations']
        costs = obj['costs']
        rate = obj['rate']
        m = obj['m']
        mini = obj['mini']
    # 如果有迁移，则从旧网络中获取已经学习过的参数
    elif enable_transfor and os.path.exists(transforfile):
        transfordata = load_params_from_old_network(transforfile)
        g = n.parse(code)
        for node in transfordata :
            if node['name'] in ['W11', 'W12', 'W21', 'W22', 'W31', 'W32', 'W41', 'W42', 'W51', 'W52'] :
                g.setData(node['name'], np.array(node['payload'], dtype=np.float32))
    else :
        g = n.parse(code)

    # 加载所有样本
    print("Samples loading...")
    X = getX(m)
    Y = getY(m)
    batches, train_X, train_Y, test_X, test_Y =  process_samples(X, Y, mini)
    print("%d samples loaded." %(X.shape[0]))

    allParams = ['W11', 'W12', 'W21', 'W22', 'W31', 'W32', 'W41', 'W42', 'W51', 'W52', 'W6', 'b6', 'W7', 'b7', 'W8', 'b8']
    
    # 添加学习更新方法
    for name in allParams :
        g.setUpdateFunc(name, erud.upf.adam(rate))


    tic = time.time()

    # 学习
    for i in range (num_iterations - num_over_iterations) :
        for b in batches :
            g.setData('X', b[0])
            g.setData('Y', b[1])

            g.fprop()
            g.bprop()
        
        cost = g.getData('J')
        costs.append(cost)

        if (i + num_over_iterations + 1) % 1 == 0 :
            # 缓存学习结果
            erud.nous_exports(n, cachefile, {
                'costs' : costs,
                'num_over_iterations' : 1 + i + num_over_iterations,
                'num_iterations' : num_iterations,
                'rate' : rate,
                'm' : m,
                'mini' : mini,
            })
            print("Cost after iteration {}: {}.".format(i + num_over_iterations + 1, cost))

        if (i + num_over_iterations + 1) % 1 == 0 :
            # 多次备份缓存
            erud.nous_exports(n, path + '/cache-bk-' + str(i + num_over_iterations + 1) + '.json', {
                'costs' : costs,
                'num_over_iterations' : 1 + i + num_over_iterations,
                'num_iterations' : num_iterations,
                'rate' : rate,
                'm' : m,
                'mini' : mini,
            })
    print("Cost after iteration {}: {}.".format(num_iterations, cost))

    toc = time.time()


    print(g.tableTimespend())
    print("Cost of time is %fs." % ((toc - tic)))


    gtest = erud.nous(
    '''
    X ->

        conv2d_v3_same(1) W11 -> batchnorm2d(0) as BN11 -> mul G11 add T11 -> relu ->
        conv2d_v3_same(1) W12 -> batchnorm2d(0) as BN12 -> mul G12 add T12 -> relu ->
        max_pool_v3(2, 2, 2) ->

        conv2d_v3_same(1) W21 -> batchnorm2d(0) as BN21 -> mul G21 add T21 -> relu ->
        conv2d_v3_same(1) W22 -> batchnorm2d(0) as BN22 -> mul G22 add T22 -> relu ->
        max_pool_v3(2, 2, 2) ->

        conv2d_v3_same(1) W31 -> batchnorm2d(0) as BN31 -> mul G31 add T31 -> relu ->
        conv2d_v3_same(1) W32 -> batchnorm2d(0) as BN32 -> mul G32 add T32 -> relu ->
        max_pool_v3(2, 2, 2) ->

        conv2d_v3_same(1) W41 -> batchnorm2d(0) as BN41 -> mul G41 add T41 -> relu ->
        conv2d_v3_same(1) W42 -> batchnorm2d(0) as BN42 -> mul G42 add T42 -> relu ->
        max_pool_v3(2, 2, 2) ->

        conv2d_v3_same(1) W51 -> batchnorm2d(0) as BN51 -> mul G51 add T51 -> relu ->
        conv2d_v3_same(1) W52 -> batchnorm2d(0) as BN52 -> mul G52 add T52 -> relu ->
        max_pool_v3(2, 2, 2) ->

    flatten ->

        matmul W6 add b6 -> relu ->

        matmul W7 add b7 -> relu ->

        matmul W8 add b8 ->

    max_index(1) -> accuracy Y -> J:$$
    '''
    ).parse()


    # 迁移参数
    for name in allParams :
        gtest.setData(name, g.getData(name))



    if m > 100 :
        # 分批计算训练集精度（内存不足...）
        accs = []
        sum_accs = int(math.floor(4521 * 3 * 0.8 / 100))
        for i in range(sum_accs) :
            gtest.setData('X', train_X[100*i:100*(i+1), :, :, :])
            gtest.setData('Y', train_Y[100*i:100*(i+1)])

            gtest.fprop()
            accs.append(float(gtest.getData('J')))

        avg_accs = sum(accs) / sum_accs
        print('train accuracy: %s' %(avg_accs))
    else :
        gtest.setData('X', train_X)
        gtest.setData('Y', train_Y)

        gtest.fprop()
        print('train accuracy: %s' %(gtest.getData('J')))

    # 计算测试集精度


    if m > 100 :
        # 分批计算训练集精度（内存不足...）
        accs = []
        sum_accs = int(math.floor(4521 * 3 * 0.2 / 100))
        for i in range(sum_accs) :
            gtest.setData('X', test_X[100*i:100*(i+1), :, :, :])
            gtest.setData('Y', test_X[100*i:100*(i+1)])

            gtest.fprop()
            accs.append(float(gtest.getData('J')))

        avg_accs = sum(accs) / sum_accs
        print('test accuracy: %s' %(avg_accs))
    else :
        gtest.setData('X', test_X)
        gtest.setData('Y', test_Y)

        gtest.fprop()

        print('test accuracy: %s' %(gtest.getData('J')))


stocking_classification()



