from matplotlib import pyplot as plt
import cv2
import sys

depth_map = cv2.imread(sys.argv[1])
img = cv2.imread(sys.argv[2])
fig, ax = plt.subplots(nrows=1, ncols=2)
plt.subplot(1, 2, 1)
plt.imshow(depth_map)
plt.subplot(1, 2, 2)
plt.imshow(img)
plt.show(block=False)
plt.pause(1000)
plt.close()
