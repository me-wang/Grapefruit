# 自定义测试图像

import cv2
import numpy as np
img = np.zeros((300, 500), dtype=np.uint8)

img[100:200, 280:320] = 255

cv2.imshow("img", img)
print("img的shape = \n", img.shape)
cv2.imwrite("D:/python_workplace/OpenCV/img1.png", img)
print("保存成功了、。。。")
cv2.waitKey()
cv2.destroyAllWindows()