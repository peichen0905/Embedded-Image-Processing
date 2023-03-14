import cv2
import numpy as np
import time

start = time.time()
img = cv2.imread("happy.jpg")
image = cv2.resize(img, (256, 512), interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)[1]

kernel = np.ones((3, 3), np.uint8)
morph_gradient = cv2.morphologyEx(threshold, cv2.MORPH_GRADIENT, kernel)

end = time.time()

cv2.imshow('original1', image)
cv2.imshow('threshold', threshold)
cv2.imshow('morph_gradient', morph_gradient)

print("執行時間：%f 秒" % (end - start))
cv2.waitKey(0)