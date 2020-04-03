# 人脸识别
import cv2
import numpy as np

images = []
images.append(cv2.imread("D:\python_workplace\Grapefruit\cv\images\hjh.jpg", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("D:\python_workplace\Grapefruit\cv\images\hjh1.jpg", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("D:\python_workplace\Grapefruit\cv\images\wzx.jpg", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("D:\python_workplace\Grapefruit\cv\images\wzx1.jpg", cv2.IMREAD_GRAYSCALE))
labels = [0, 0, 1, 1]
recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.train(images, np.array(labels))
predict_image = cv2.imread("D:\python_workplace\Grapefruit\cv\images\wzx3.jpg", cv2.IMREAD_GRAYSCALE)
labels, confidence = recognizer.predict(predict_image)
print("labels= ", labels)
print("confidence = ", confidence)

