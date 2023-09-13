import json
import random
import cv2
import os
import numpy as np


def process_image(file) :
    """
    处理图片，将所有图片按比例缩放并裁剪成（128,192）大小

    1. 如果横着的图片就竖起来
    2. 缩小到刚好能把（128,192）大小的块包裹起来的等比大小图片
    3. 裁剪成（128,192）大小并存储
    """
    img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
    rate = 4. / 6
    # 如果宽高比例大于1，即图片是PC图片，则转置
    (h, w, c) = img.shape
    if h < w :
        img = cv2.transpose(img)
    
    # getRectSubPix不支持四通道（带alpha通道）图片
    img = img[:,:,0:3]
    
    (h, w, c) = img.shape
    cr = 1. * w / h
    
    # 如果图片矮胖，缩放至高度相同，居中裁剪
    if cr < rate :
        r = 128. / w
        img = cv2.resize(img, None, fx = r, fy = r)
        (h, w, c) = img.shape
        img = cv2.getRectSubPix(img, (128, 192), (int(w / 2.), int(h / 2.)))
    # 如果图片高瘦，缩放至宽度相同，居中裁剪
    else :
        r = 192. / h
        img = cv2.resize(img, None, fx = r, fy = r)
        (h, w, c) = img.shape
        img = cv2.getRectSubPix(img, (128, 192), (int(w / 2.), int(h / 2.)))
    
    return cv2.transpose(img).tolist()


def verify_from_train(idx) :

    # img = process_image( 'D:\Study\proj\static\proj-samples-enhance\\334.jpg' )
    img = cv2.imdecode(np.fromfile( 'D:\Study\proj\static\proj-samples-enhance\\%s.jpg' % (str(idx)), dtype=np.uint8), -1)

    labelfile = "D:\Study\proj\static\labelfile-enhance.json"
    with open (labelfile, 'r', encoding='utf-8') as f:
        labels = json.loads(f.read())



    import sys
    sys.path.append('D:\Study\erud')
    import erud

    cachefile = __file__[:__file__.rfind('\\')] + '/cache.json'
    n, obj = erud.nous_imports(cachefile)
    g = n.g



    gtest = erud.nous(
    '''
    X ->

        conv2d_v3_same(1) W11 -> batchnorm2d -> relu ->
        conv2d_v3_same(1) W12 -> batchnorm2d -> relu ->
        max_pool_v3(2, 2, 2) ->

        conv2d_v3_same(1) W21 -> batchnorm2d -> relu ->
        conv2d_v3_same(1) W22 -> batchnorm2d -> relu ->
        max_pool_v3(2, 2, 2) ->

        conv2d_v3_same(1) W31 -> batchnorm2d -> relu ->
        conv2d_v3_same(1) W32 -> batchnorm2d -> relu ->
        max_pool_v3(2, 2, 2) ->

        conv2d_v3_same(1) W41 -> batchnorm2d -> relu ->
        conv2d_v3_same(1) W42 -> batchnorm2d -> relu ->
        max_pool_v3(2, 2, 2) ->

        conv2d_v3_same(1) W51 -> batchnorm2d -> relu ->
        conv2d_v3_same(1) W52 -> batchnorm2d -> relu ->
        max_pool_v3(2, 2, 2) ->

    flatten ->

        matmul W6 add b6 -> relu ->

        matmul W7 add b7 -> relu ->

        matmul W8 add b8 -> A:$$

    A -> max_index(1) -> J:$$
    '''
    ).parse()

    allParams = ['W11', 'W12', 'W21', 'W22', 'W31', 'W32', 'W41', 'W42', 'W51', 'W52', 'W6', 'b6', 'W7', 'b7', 'W8', 'b8']

    # 迁移参数
    for name in allParams :
        gtest.setData(name, g.getData(name))
    X = np.array([img])
    enum = {
        0: "无",
        1: "黑丝",
        2: "白丝",
        3: "花丝",
        4: "裸足",
    }
    gtest.setData('X', X)
    gtest.fprop()
    print('预测 %s 但是 %s' % (enum[gtest.getData('J')[0,0]], enum[labels[idx - 1]]))

    print()
    print(gtest.getData('A'))
    print(np.eye(5)[labels[idx - 1]])


for i in range(10) :
    idx = random.randint(1, 4521 * 3 + 1)
    print('图片 %s' %(idx))
    verify_from_train(idx)