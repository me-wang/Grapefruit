# 文档扫描OCR识别
# step1:边缘检测
# step2：获取轮廓
# step3：变换
# step4：OCR
import cv2
import numpy as np
screenCnt = []
def order_point(pts):
    # 一共四个坐标
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_point(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32"
    )
    print("111111")
    M =cv2.getPerspectiveTransform(rect, dst)
    print(M)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

image = cv2.imread("E:/pycharmWorkplace/Grapefruit/cv/images/OCR1.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()

image = resize(orig, height=500)
# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
# cv2.imwrite("E:/pycharmWorkplace/OpenCV/img_prepare/edged.jpg", edged)

# 显示预处理结果
# print("step1:边缘检测")
# cv2.imshow("img", image)
# cv2.imshow("edged", edged)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 检测轮廓
cnts, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # 面积排序，前五个
# 遍历轮廓
for c in cnts:
    # 计算轮廓近似
    peri = cv2.arcLength(c, True)
    # c表示输入的点集
    # epsilon表示从原始轮廓到近似轮廓的最大距离，他是一个准确参数
    # True表示封闭的
    approx = cv2.approxPolyDP(c, 0.02*peri, True)

    # 4个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        break

# 展示结果
print("Steep2 :获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)


# 二值处理
wraped1 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

t, ref = cv2.threshold(wraped1, 180, 255, cv2.THRESH_BINARY)

ref2 = resize(ref, height=650)

def ref_pro(ref):
    ref1 = resize(ref, height=650)  # 上50 左 5 右 4  下 3   (650,396)

    for i in range(70):
        for j in range(396):
            ref1[i, j] = 255
    for i in range(650):
        for j in range(5):
            ref1[i, j] = 255
    for i in range(640, 650):
        for j in range(396):
            ref1[i, j] = 255
    for i in range(650):
        for j in range(390, 396):
            ref1[i, j] = 255
    return ref1

ref1 = ref_pro(ref)

print("Step 3: 变换")
cv2.imshow("Original", resize(orig, height=650))
cv2.imshow("Scanned2", ref2)
cv2.imshow("Scanned", ref1)
cv2.waitKey(0)
cv2.destroyAllWindows()
