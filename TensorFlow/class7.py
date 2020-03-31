# 边缘检测


import cv2

import numpy as np

img = cv2.imread("D:/python_workplace/OpenCV/test.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=0.5, fy=0.5)
imgSobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
imgSobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
imgSobel_x = cv2.convertScaleAbs(imgSobel_x)
imgSobel_y = cv2.convertScaleAbs(imgSobel_y)
imgSobel = cv2.addWeighted(imgSobel_x, 0.5, imgSobel_y, 0.5, 0)

imgScharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
imgScharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
imgScharr_x = cv2.convertScaleAbs(imgScharr_x)
imgScharr_y = cv2.convertScaleAbs(imgScharr_y)
imgScharr = cv2.addWeighted(imgScharr_x, 0.5, imgScharr_y, 0.5, 0)

cv2.imshow("imgSobel", imgSobel)
cv2.imshow("imgScharr", imgScharr)
cv2.waitKey()
cv2.destroyAllWindows()