import os
import time
import math
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('D:\Study\erud')
import erud



def load_mnist_train(images_path, labels_path) :

    with gzip.open(images_path, 'rb') as fimg :
        magic, num, rows, cols = struct.unpack('>IIII', fimg.read(16))
        images = np.frombuffer(fimg.read(), dtype=np.uint8).reshape((num, 28, 28, 1))

    with gzip.open(labels_path, 'rb') as flab :
        magic, num = struct.unpack('>II', flab.read(8))
        labels = np.frombuffer(flab.read(), dtype=np.uint8)
    
    return images, labels, num, rows, cols

def show_mini_imgs (images, labels) :
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    assert images[0].shape == (28, 28, 1)
    for i in range(30) :
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i].reshape((28, 28)), cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(np.argmax(labels[i])))
    plt.show()

def process_samples(images, labels, s, random = False) :
    m = images.shape[0]
    
    # np.random.seed(1)
    if random :
        permutation = list(np.random.permutation(m))
    else :
        permutation = [i for i in m]

    shuffled_X = images[permutation, :, :, :] / 255. - .5
    shuffled_Y = labels[permutation]
    shuffled_Y_one_hot = np.eye(10)[shuffled_Y.reshape(-1)]

    n = math.floor(m / s)
    batches = []

    for k in range(0, n) :
        mX = shuffled_X[k * s : (k + 1) * s, :, :, :]
        mY = shuffled_Y_one_hot[k * s : (k + 1) * s, :]

        batches.append((mX, mY))
    
    if m % s != 0:
        mX = shuffled_X[m - (m % s):, :, :, :]
        mY = shuffled_Y_one_hot[m - (m % s):, :]

        batches.append((mX, mY))
    
    return batches, shuffled_X, shuffled_Y

def init_cg(cachefile, num_iterations = 120, num_over_iterations = 0, m = 60000, mini = 1024) :
    # 没有batchnorm2d的
    # lenet-5改进
    code = """
    X:(1, 28, 28, 1) ->

        conv2d_v3_same W1:xavier((5, 5, 1, 6), 25):norm(0.001) add b1:(6):norm(0.001) -> relu -> 

        ########## 变为(?, 28, 28, 6) 

        max_pool_v3(2, 2, 2) ->

        ########## 变为(?, 14, 14, 6) 

        conv2d_v3 W2:xavier((5, 5, 6, 16), 25):norm(0.001) add b2:(16):norm(0.001) -> relu ->

        ########## 变为(?, 10, 10, 16) 

        max_pool_v3(2, 2, 2) ->

        ########## 变为(?, 5, 5, 16) 

    flatten ->

        ########## 变为(?, 400) 

        matmul W3:he((400, 120), 400):norm(0.001) add b3:(120):norm(0.001) -> relu ->

        matmul W4:he((120, 84), 120):norm(0.001) add b4:(84):norm(0.001) -> relu ->

        matmul W5:he((84, 10), 84):norm(0.001) add b5:(10):norm(0.001) ->

    softmax_cross_entropy(1) Y:(1, 10) -> cost -> J:$$
    """
    # 有batchnorm2d的
    # vgg-16
    """
    X:(1, 28, 28, 1) ->

        conv2d_v3_same W11:xavier((3, 3, 1, 4), 9):adam(0.0002) -> 
        batchnorm2d as BN11 -> 
        mul G11:randn(4):adam(0.0002) add T11:(4):adam(0.0002) -> 
        relu ->
        
        conv2d_v3_same W12:xavier((3, 3, 4, 4), 9):adam(0.0002) -> 
        batchnorm2d as BN12 -> 
        mul G12:randn(4):adam(0.0002) add T12:(4):adam(0.0002) -> 
        relu ->

        max_pool_v3(2, 2, 2) ->

        ########## 变为(?, 14, 14, 4) 

        conv2d_v3_same W21:xavier((3, 3, 4, 8), 9):adam(0.0002) ->
        batchnorm2d as BN21 ->
        mul G21:randn(8):adam(0.0002) add T21:(8):adam(0.0002) ->
        relu ->

        conv2d_v3_same W22:xavier((3, 3, 8, 8), 9):adam(0.0002) ->
        batchnorm2d as BN22 ->
        mul G22:randn(8):adam(0.0002) add T22:(8):adam(0.0002) ->
        relu ->

        max_pool_v3(2, 2, 2) ->

        ########## 变为(?, 7, 7, 8) 

        conv2d_v3_same W31:xavier((3, 3, 8, 16), 9):adam(0.0002) ->
        batchnorm2d as BN31 ->
        mul G31:randn(16):adam(0.0002) add T31:(16):adam(0.0002) ->
        relu ->

        conv2d_v3_same W32:xavier((3, 3, 16, 16), 9):adam(0.0002) ->
        batchnorm2d as BN32 ->
        mul G32:randn(16):adam(0.0002) add T32:(16):adam(0.0002) ->
        relu ->

        ########## max_pool_v3(2, 2, 2) ->

        ########## 变为(?, 7, 7, 16) 

        conv2d_v3_same W41:xavier((3, 3, 16, 32), 9):adam(0.0002) ->
        batchnorm2d as BN41 ->
        mul G41:randn(32):adam(0.0002) add T41:(32):adam(0.0002) ->
        relu ->

        conv2d_v3_same W42:xavier((3, 3, 32, 32), 9):adam(0.0002) ->
        batchnorm2d as BN42 ->
        mul G42:randn(32):adam(0.0002) add T42:(32):adam(0.0002) ->
        relu ->

        ########## max_pool_v3(2, 2, 2) ->

        ########## 变为(?, 7, 7, 32) 
        

    flatten ->

        matmul W6:xavier((1568, 128), 1568):adam(0.0002) add b6:(128):adam(0.0002) -> relu ->

        matmul W7:xavier((128, 32), 128):adam(0.0002) add b7:(32):adam(0.0002) -> relu ->

        matmul W8:xavier((32, 10), 32):adam(0.0002) add b8:(10):adam(0.0002) ->

    softmax_cross_entropy(1) Y:(1, 10) -> cost -> J:$$
    """

    costs = []

    if os.path.exists(cachefile) :
        n, obj = erud.imports(cachefile)
        g = n.g
        num_iterations = obj['num_iterations']
        num_over_iterations = obj['num_over_iterations']
        costs = obj['costs']
        m = obj['m']
        mini = obj['mini']
    else :
        n = erud.nous()
        g = n.parse(code)
    
    return g, n, num_iterations, num_over_iterations, costs, m, mini


def init_test_cg(cachefile) :

    # 没有batchnorm2d的
    code = """
    X ->

        conv2d_v3_same W1 add b1 -> relu ->

        max_pool_v3(2, 2, 2) ->

        conv2d_v3 W2 add b2 -> relu ->

        max_pool_v3(2, 2, 2) ->

    flatten ->

        matmul W3 add b3 -> relu ->

        matmul W4 add b4 -> relu ->

        matmul W5 add b5 ->

    max_index(1) -> accuracy Y -> J:$$
    """
    # 有batchnorm2d的
    """
    X ->

        conv2d_v3_same W11 -> batchnorm2d(0) as BN11 -> mul G11 add T11 -> relu ->
        conv2d_v3_same W12 -> batchnorm2d(0) as BN12 -> mul G12 add T12 -> relu ->
        max_pool_v3(2, 2, 2) ->

        ########## 变为(?, 14, 14, 4) 

        conv2d_v3_same W21 -> batchnorm2d(0) as BN21 -> mul G21 add T21 -> relu ->
        conv2d_v3_same W22 -> batchnorm2d(0) as BN22 -> mul G22 add T22 -> relu ->
        max_pool_v3(2, 2, 2) ->

        ########## 变为(?, 7, 7, 8) 

        conv2d_v3_same W31 -> batchnorm2d(0) as BN31 -> mul G31 add T31 -> relu ->
        conv2d_v3_same W32 -> batchnorm2d(0) as BN32 -> mul G32 add T32 -> relu ->

        ########## max_pool_v3(2, 2, 2) ->

        ########## 变为(?, 7, 7, 16) 

        conv2d_v3_same W41 -> batchnorm2d(0) as BN41 -> mul G41 add T41 -> relu ->
        conv2d_v3_same W42 -> batchnorm2d(0) as BN42 -> mul G42 add T42 -> relu ->

        ########## max_pool_v3(2, 2, 2) ->

        ########## 变为(?, 7, 7, 32) 
        

    flatten ->

        matmul W6 add b6 -> relu ->

        matmul W7 add b7 -> relu ->

        matmul W8 add b8 ->

    max_index(1) -> accuracy Y -> J:$$
    """

    ntest = erud.nous(code)
    ntest.parse()
    erud.transfer(ntest, cachefile)

    gtest = ntest.g

    return gtest, ntest
    




if __name__ == '__main__' :
    # test
    path = __file__[:__file__.rfind('\\')]
    images_path = path + '/static/mnist/train-images-idx3-ubyte.gz'
    labels_path = path + '/static/mnist/train-labels-idx1-ubyte.gz'
    images, labels, num, rows, cols = load_mnist_train(images_path, labels_path)

    images_test_path = path + '/static/mnist/t10k-images-idx3-ubyte.gz'
    labels_test_path = path + '/static/mnist/t10k-labels-idx1-ubyte.gz'
    test_imgs, test_labels, _, _, _ = load_mnist_train(images_test_path, labels_test_path)

    assert num == 60000
    assert rows == 28
    assert cols == 28
    assert images.shape == (60000, 28, 28, 1)
    assert labels.shape == (60000,)
    m = 60000
    mini = 2048

    batches, trainX, trainY = process_samples(images[0:m], labels[0:m], mini, True)
    # assert batches[0][0].shape == (1024, 28, 28, 1)
    # assert batches[0][1].shape == (1024, 10)
    # show_mini_imgs(batches[0][0], batches[0][1])

    path = __file__[:__file__.rfind('\\')]
    cachefile = path + '/mnist-cache.json'

    g, n, num_iterations, num_over_iterations, costs, m, mini = init_cg(cachefile, m=m, mini=mini)

    tic = time.time()

    cost = 0

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
            erud.exports(n, cachefile, {
                'costs' : costs,
                'num_over_iterations' : 1 + i + num_over_iterations,
                'num_iterations' : num_iterations,
                'm' : m,
                'mini' : mini,
            })
            print("Cost after iteration {}: {}.".format(i + num_over_iterations + 1, cost))

        if (i + num_over_iterations + 1) % 20 == 0 :
            # 多次备份缓存
            erud.exports(n, path + '/mnist-cache-bk-' + str(i + num_over_iterations + 1) + '.json', {
                'costs' : costs,
                'num_over_iterations' : 1 + i + num_over_iterations,
                'num_iterations' : num_iterations,
                'm' : m,
                'mini' : mini,
            })
    print("Cost after iteration {}: {}.".format(num_iterations, cost))

    toc = time.time()


    print(g.tableTimespend())
    print("Cost of time is %fs." % ((toc - tic)))

    gtest, _ = init_test_cg(cachefile)

    gtest.setData('X', trainX)
    gtest.setData('Y', trainY)
    gtest.fprop()

    print('train accuracy: %s' %(gtest.getData('J')))

    gtest.setData('X', test_imgs / 255.)
    gtest.setData('Y', test_labels)
    gtest.fprop()

    print('test accuracy: %s' %(gtest.getData('J')))
