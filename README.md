# ip
# 1.develop a program to display grayscale image using read and write operation
In digital photography, computer-generated imagery, and colorimetry, a grayscale or image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.
To convert an image to grayscale in any of the Microsoft Office suite apps, right-click it and select Format Picture from the context menu . This will open an image editing panel on the right. Go to the Picture tab (the very last one). Expand the Picture Color options, and click the little dropdown next to the Presets for Color Saturation.
import cv2 as c
import numpy as np
image = c.imread("rose.jpg")
gray  = c.cvtColor(image,c.COLOR_BGR2GRAY)
c.imshow("Fist Lab",gray)
cv2.imwrite("rajath.jpg",grayimg)
c.waitKey(0)
c.destroyAllWindows()
output

# 2
