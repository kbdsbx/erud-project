import cv2
import os
import numpy as np
import json
import math
import time


# img = cv2.imread('./img/0003.jpg')
# (w, h, c) = img.shape

# resize to (227 x 227 x 3)

def calcuate_average_proportion () :
    path = "D:\OneDrive\Images\XP16"

    sw = 0
    sh = 0
    i = 0

    for f in os.listdir(path) :
        i+=1
        if i % 100 == 0 :
            print('Computing %d.' % (i))
        pathname = os.path.join(path, f)
        if os.path.isfile(pathname) and (pathname.endswith('jpg') or pathname.endswith('png')):
            img = cv2.imdecode(np.fromfile(pathname, dtype=np.uint8), -1)
            (w, h, c) = img.shape
            if w > h :
                sw += w
                sh += h
            else :
                sw += h
                sh += w
                

    t = 1. * sw / sh

    print('The best proportion is %.4f in %d images(%d, %d).' % (t, i, sw, sh))


# 缩放的代码
# cv2.resize(img, (128, 192))
# cv2.resize(img, None, fx = 0.3, fy = 0.3)

# 剪裁的代码
# cv2.getRectSubPix(img, (w, h), (x, y))

# 保存文件的代码
# cv2.imwrite("src", img)

def process_all_images () :
    """
    处理图片，将所有图片按比例缩放并裁剪成（128,192）大小

    1. 如果横着的图片就竖起来
    2. 缩小到刚好能把（128,192）大小的块包裹起来的等比大小图片
    3. 裁剪成（128,192）大小并存储
    """
    path = "D:\Study\proj\static\proj-samples"
    outpath = "D:\Study\proj\static\proj-caches"
    origpath = "D:\Study\proj\static\proj-samples-origin"
    rate = 4. / 6
    i = 0

    for f in os.listdir(path) :
        # if i % 100 == 0 :
            # print('Computing %d.' % (i))
        pathname = os.path.join(path, f)

        if os.path.isfile(pathname) and (pathname.endswith('jpg') or pathname.endswith('png') or pathname.endswith('jpeg')):

            i+=1
            
            img = cv2.imdecode(np.fromfile(pathname, dtype=np.uint8), -1)
            org = img
            (h, w, c) = img.shape
            # 如果宽高比例大于1，即图片是PC图片，则转置
            print('%d, %d, %d, %s' % (i, w, h, pathname))
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
            
            cv2.imwrite(os.path.join(outpath, str(i) + '.jpg'), img)
            cv2.imwrite(os.path.join(origpath, str(i) + '.jpg'), org)


# process_all_images()

