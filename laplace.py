#!/usr/bin/env python
# encoding=gbk

'''
Laplacian����
��OpenCV-Python�У�Laplace���ӵĺ���ԭ�����£�
dst = cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])  
��һ����������Ҫ�����ͼ��
�ڶ���������ͼ�����ȣ�-1��ʾ���õ�����ԭͼ����ͬ����ȡ�Ŀ��ͼ�����ȱ�����ڵ���ԭͼ�����ȣ�
dst���ý����ˣ�
ksize�����ӵĴ�С������Ϊ1��3��5��7��Ĭ��Ϊ1��
scale�����ŵ����ı���������Ĭ�������û������ϵ����
delta��һ����ѡ������������ӵ����յ�dst�У�ͬ����Ĭ�������û�ж����ֵ�ӵ�dst�У�
borderType���ж�ͼ��߽��ģʽ���������Ĭ��ֵΪcv2.BORDER_DEFAULT��
'''
 
import cv2
import numpy as np 
 
img=cv2.imread('lenna.png',0)
 
#Ϊ���ý���������������ksize��Ϊ3��
gray_lap=cv2.Laplacian(img,cv2.CV_16S,ksize=3)#��ʽ����
dst=cv2.convertScaleAbs(gray_lap)
 
cv2.imshow('laplacian',dst)
 
cv2.waitKey(0)
cv2.destroyAllWindows()
