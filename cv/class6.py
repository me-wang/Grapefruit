import cv2
# 均值滤波

img = cv2.imread("D:/python_workplace/OpenCV/test.jpg", 1)
img_blur = cv2.blur(img, (10, 10))
cv2.imshow("img", img)
cv2.imshow("img_blur", img_blur)
cv2.waitKey()
cv2.destroyAllWindows()