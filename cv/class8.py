# 图像大小测试

import cv2

img = cv2.imread("D:/python_workplace/OpenCV/test.jpg", 1)
cv2.imshow("img", img)
print("img的shape = \n", img.shape)
cv2.waitKey()
cv2.destroyAllWindows()