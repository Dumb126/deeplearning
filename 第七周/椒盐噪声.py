# 给一幅图加上椒盐噪声的处理步骤
"""
    1、指定信噪比SNR，其取值范围为[0-1]
    2、计算总像素数目SP,得到要加噪声的像素数目NP = SP *（1-SNR）
    3、随机获取要加噪声的每个像素的位置p[i,j]
    4、指定像素值为255或者0
    5、重复3、4步骤
"""
import numpy as np
import cv2
from numpy import shape
import random
def fun1(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255

    return NoiseImg

img = cv2.imread("lenna.png", 0)
img1 = fun1(img, 0.2)
cv2.imwrite("lenna_PepperandSalt.png", img1)

img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', img2)
cv2.imshow("lenna_PepperandSalt", img1)
cv2.waitKey(0)