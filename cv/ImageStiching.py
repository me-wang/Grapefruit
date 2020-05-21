import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取拼接图片
imageA = cv2.imread("./images/left_01.png", 0)
imageB = cv2.imread("./images/right_01.png", 0)


# 显示图片
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# cv_show("img1", imageA)
# cv_show("img2", imageB)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(imageA, None)
kp2, des2 = sift.detectAndCompute(imageB, None)
bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)
good = []

for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img = cv2.drawMatchesKnn(imageA, kp1, imageB, kp2, matches[:10], good, None, flags=2)
cv_show("img", img)