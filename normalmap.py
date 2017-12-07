#! /usr/bin/python

import cv2
import sys
import os
import numpy as np

print("loading image ", sys.argv[1])

name = os.path.splitext(sys.argv[1])[0]


if len(sys.argv) == 2:
	strength = 1.0
else:
	strength = float(sys.argv[2])

diffuse = cv2.imread(sys.argv[1])

average = cv2.cvtColor(diffuse, cv2.COLOR_RGB2GRAY)

average = cv2.normalize(average.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

#average = cv2.GaussianBlur(average, (3,3), 0)


sobelx = cv2.Sobel(average, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(average, -1, 0, 1, ksize=3)

ones = np.ones(average.shape)

n = np.stack((strength*sobelx,strength*sobely, ones), axis=2)

norm =  np.linalg.norm(n, axis=2, keepdims=True)

n = n / norm

n = n*0.5+0.5

normalmap = np.stack((n[:, : , 2], n[:, :, 1], 1.0 - n[: ,: ,0]), axis=2)

cv2.imshow("output", normalmap)


cv2.waitKey(0)
cv2.destroyAllWindows()

output = np.array(normalmap*255, dtype=np.uint8)

output = cv2.cvtColor(output, cv2.CV_8U)

cv2.imwrite(name+"_norm.jpg", output)
