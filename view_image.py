from matplotlib import pyplot as plt
import cv2

img = cv2.imread("out.png")
plt.imshow(img)
plt.pause(30)
plt.close()
