# 图像梯度处理

import cv2
import numpy as np
# Sobel算子 Gx=[[-1,0,1],[-2,0,2],[-1,0,1]] Gy=[[-1,-2,-1],[0,0,0],[1,2,1]]

img = cv2.imread("E:/pycharmWorkplace/Grapefruit/cv/images/lena_test.jpg", 0)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv2.imshow("sobelxy", sobelxy)


# Scharr算子 Gx=[[-3,0,3],[-10,0,10],[-3,0,3]] Gy=[[-3,10,3],[0,0,0],[-3,10,3]]
# 检测边缘比较细致
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
cv2.imshow("scharrxy", scharrxy)


# laplacian算子 G=[[0,1,0],[1,-4,1],[0,1,0]]
# 需要和其他算法配合使用
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
cv2.imshow("laplacian", laplacian)

cv2.waitKey()
cv2.destroyAllWindows()