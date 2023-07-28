import random
import cv2
import os
import numpy as np

def process_image(img, file) :
    """
    处理图片，将所有图片按比例缩放并裁剪成（128,192）大小

    1. 如果横着的图片就竖起来
    2. 缩小到刚好能把（128,192）大小的块包裹起来的等比大小图片
    3. 裁剪成（128,192）大小并存储
    """
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
        img = cv2.getRectSubPix(img, (128, 192), (int(w / 2.), int(h / 2.) + random.randrange(-10, 10)))
    # 如果图片高瘦，缩放至宽度相同，居中裁剪
    else :
        r = 192. / h
        img = cv2.resize(img, None, fx = r, fy = r)
        (h, w, c) = img.shape
        img = cv2.getRectSubPix(img, (128, 192), (int(w / 2.) + random.randrange(-10, 10), int(h / 2.)))
    
    cv2.imwrite(file, img)

def process_all_images () :
    """
    图片增强，相同标签的图片放大多倍

    1. 缩放图片(1.1, 1.3)
    2. 旋转180度并放大1.01倍
    """
    origpath = "D:\Study\proj\static\proj-samples-origin"
    enhancepath = "D:\Study\proj\static\proj-samples-enhance"
    m = 4521

    for p in range(m) :
        i = p+1
        # if i % 100 == 0 :
            # print('Computing %d.' % (i))
        pathname = os.path.join(origpath, str(i) + '.jpg')

        if os.path.isfile(pathname) :

            img = cv2.imdecode(np.fromfile(pathname, dtype=np.uint8), -1)
            (h, w, c) = img.shape
            print('%d, %d, %d, %s' % (i, w, h, pathname))

            # 输出原始图片的裁剪
            process_image(img, os.path.join(enhancepath, str(i) + '.jpg'))

            # 缩放
            scale = np.float32([[1.1,0,0],[0,1.3,-30]])
            img = cv2.warpAffine(img, scale, (w, h))

            # 输出增强图片的裁剪
            process_image(img, os.path.join(enhancepath, str(i + m) + '.jpg'))

            # 旋转
            rotated = cv2.getRotationMatrix2D((w/2., h/2.), 180, 1.01)
            img = cv2.warpAffine(img, rotated, (w, h))

            # 输出增强图片的裁剪
            process_image(img, os.path.join(enhancepath, str(i + m * 2) + '.jpg'))


process_all_images()

