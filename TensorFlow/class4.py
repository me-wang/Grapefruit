import cv2
import numpy as np

img = np.zeros((5, 5), dtype=np.uint8)
img[0:6, 0:6] = 123
img[2:6, 2:6] = 126
print("img = \n", img)
cv2.imshow("img", img)
t1, thd =  cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print("thd = \n", thd)
cv2.imshow("thd", thd)
t2, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("otus = \n", otsu)
cv2.imshow("otsu", otsu)
cv2.waitKey()
cv2.destroyAllWindows()
