import numpy as np
import cv2


img = cv2.imread("D:/python_workplace/OpenCV/test.jpg", 0)
img[100:200, 100:200] = 255
cv2.imshow("img", img)

cv2.waitKey()
cv2.destroyAllWindows()