import numpy as np
import cv2
from matplotlib import pyplot as plt


# 自定义 LBP 函数
def lbp(image, radius, neighbors):
    # 计算 LBP 算法的特征
    lbp_image = np.zeros_like(image)
    for i in range(radius, image.shape[0] - radius):
        for j in range(radius, image.shape[1] - radius):
            center = image[i, j]
            lbp_code = 0
            for k in range(neighbors):
                angle = k * (2 * np.pi / neighbors)
                x = i + int(round(radius * np.cos(angle)))
                y = j - int(round(radius * np.sin(angle)))
                if image[x, y] >= center:
                    lbp_code += 2 ** k
            lbp_image[i, j] = lbp_code

    return lbp_image


# 读取图像
ia = cv2.imread('road.jpg')
im = cv2.imread('road2.jpg', cv2.IMREAD_GRAYSCALE)
ima = cv2.resize(ia, (600, 400), interpolation=cv2.INTER_AREA)
img = cv2.resize(im, (600, 400), interpolation=cv2.INTER_AREA)

x1, y1, w1, h1 = 200, 300, 20, 20
x2, y2, w2, h2 = 300, 300, 20, 20

# 框出感兴趣的区域
cv2.rectangle(ima, (x1, y1), (x1 + w1, y1 + h1), (90, 0, 180), 2)
cv2.rectangle(ima, (x2, y2), (x2 + w2, y2 + h2), (90, 0, 180), 2)
# 计算 LBP 特征
radius = 1
neighbors = 8
lbp_image = lbp(img, radius, neighbors)

hist1 = cv2.calcHist([lbp_image[y1:y1 + h1, x1:x1 + w1]], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([lbp_image[y2:y2 + h2, x2:x2 + w2]], [0], None, [256], [0, 256])
hist_dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
# 显示 LBP 特征图像

# 显示直方图
plt.subplot(2, 1, 1)
plt.plot(hist1)
plt.xlim([0, 256])
plt.title('Region 1 LBP Histogram')

plt.subplot(2, 1, 2)
plt.plot(hist2)
plt.xlim([0, 256])
plt.title('Region 2 LBP Histogram')

plt.show()

# 判断材质是否相似
threshold = 200
if hist_dist < threshold:
    print('材质相似')
else:
    print('材质不相似')
print(hist_dist)
cv2.imshow('Image', ima)
cv2.waitKey(0)
cv2.destroyAllWindows()
