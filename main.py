# -*- coding: utf-8 -*-

"""
@version: 1.0
@license: Apache Licence 
@author:  kht,cking616
@contact: cking616@mail.ustc.edu.cn
@software: PyCharm Community Edition
@file: main.py
@time: 2018/5/15
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def test_img(jpg):
    img = cv2.imread(jpg)
    sp = img.shape
    start = time.clock()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.threshold(gray, 32, 100, cv2.THRESH_BINARY, gray)
    gray = cv2.Canny(gray, 0, 0)
    gray = np.uint8(np.absolute(gray))
    cv2.GaussianBlur(gray, (7, 7), 2.5, gray, 2.5)

    """
    plt.subplot(121)
    plt.imshow(gray, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()
    """

    circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, sp[1] / 2,
                                param1=100, param2=90, minRadius=0, maxRadius=300)
    end = time.clock()
    if isinstance(circles1, np.ndarray):
        circles = circles1[0, :, :]
        circles = np.uint16(np.around(circles))
        for i in circles[:]:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5)
            cv2.circle(img, (i[0], i[1]), 2, (255, 0, 255), 10)
            cv2.rectangle(img, (i[0]-i[2], i[1]+i[2]), (i[0]+i[2], i[1]-i[2]), (255, 255, 0), 5)

        plt.subplot(121)
        plt.imshow(gray, 'gray')
        plt.xticks([])
        plt.yticks([])

        message = "Picture:" + jpg + ":"
        print(message)
        message = "Hight:" + str(sp[0]) + "  Width:" + str(sp[1])
        print(message)
        xp = sp[1] / 2 - i[0]
        yp = sp[0] / 2 - i[1]
        print("FOUP Center Position: X:", i[0], "Y:", i[1])
        message = "FOUP center offset\nDeltX:" + str(xp) + "  DeltY:" + str(yp)
        print(message)
        print("Computation cost:", end - start)
        plt.subplot(122)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.close()
    else:
        print("Don't find any FOUP!")
        input("Enter to continue")


if __name__ == '__main__':
    if not os.path.exists('img'):
        os.mkdir('img')

    print('第一步将测试图像放到img文件夹下')
    print('注意:请使用jpg格式的图片')
    input("Enter键继续")
    for (root, dirs, files) in os.walk('.\img'):
        for filename in files:
            if filename.endswith('.jpg'):
                test_img(os.path.join(root, filename))

    input("Enter退出")
