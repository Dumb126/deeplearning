# hash算法实现
import cv2
import numpy as np

# 均值哈希算法
def aHash(img):
    #  4*4 像素邻域的双三次插值
    img = cv2.resize(img,(8, 8),interpolation= cv2.INTER_CUBIC)  # interpolation 图像插值
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0   # 像素和初值为0
    hash_str = ''   # hash值初值为‘’
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1  相反为0 生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

#差值感知算法
def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素 为1  相反为0
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# hash值对比,
def cmpHash(hash1, hash2):
    n = 0
    # hash 长度不同则返回-1，代表传参出错
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        # 不相等则n+1
        if hash1[i] != hash2[i]:
            n = n+1
    return n

img1 = cv2.imread("lenna.png")
img2 = cv2.imread("lenna_noise.png")
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print("均值hash算法相似度：", n)

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print("差值哈希算法相似度为：", n)
