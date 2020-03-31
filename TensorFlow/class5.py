import cv2

# 图片阀值处理，图片缩小
img = cv2.imread("D:/python_workplace/OpenCV/test.jpg", 0)

t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


img = cv2.resize(img, None, fx=0.5, fy=0.5)
thd = cv2.resize(thd, None, fx=0.5, fy=0.5)
print("img= \n", img.shape)
print("thd= \n", thd.shape)
cv2.imshow("img", img)
cv2.imshow("thd", thd)
cv2.waitKey()
cv2.destroyAllWindows()