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


![image](https://user-images.githubusercontent.com/72590669/104425684-f8adbd80-55a6-11eb-846c-90960482e44d.png)
![image](https://user-images.githubusercontent.com/72590669/104426836-7faf6580-55a8-11eb-9fc9-0ba7d7901ada.png)

# 2.Develop a program to perform linear transformation image scaling and rotation.
Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method. It is used for manipulation of an image so that the result is more suitable than the original for a specific application.
Image scaling is a computer graphics process that increases or decreases the size of a digital image. An image can be scaled explicitly with an image viewer or editing software, or it can be done automatically by a program to fit an image into a differently sized area.
Image rotation is a common image processing routine with applications in matching, alignment, and other image-based algorithms. The input to an image rotation routine is an image, the rotation angle θ, and a point about which rotation is done.

import cv2 as c
import numpy as np
image = c.imread("rose.jpg")
gray = c.cvtColor(image,c.COLOR_BGR2RGB)
h,w = image.shape[0:2]
width = int(w * 0.5)
hight = int(h * 0.5)
res = c.resize(image,(width,hight))
c.imshow("Fist Lab",res)
c.waitKey(0)
c.destroyAllWindows()

![image](https://user-images.githubusercontent.com/72590669/104428118-14669300-55aa-11eb-98ca-867d28539de4.png)



# Scaling

import cv2 as c
import numpy as np
image = c.imread("rose.jpg")
gray = c.cvtColor(image,c.COLOR_BGR2RGB)
h,w = image.shape[0:2]
width = int(w * 3)
hight = int(h * 3)
res = c.resize(image,(width,hight))
c.imshow("Fist Lab",res)
c.waitKey(0)
c.destroyAllWindows()

# Scaling 1
![image](https://user-images.githubusercontent.com/72590669/104428118-14669300-55aa-11eb-98ca-867d28539de4.png)

# scaling 2
![image](https://user-images.githubusercontent.com/72590669/104428437-7d4e0b00-55aa-11eb-876c-2ff0d8bf51a7.png)





# Rotation

import cv2 as c

import numpy as np

image = c.imread("rose.jpg")

gray = c.cvtColor(image,c.COLOR_BGR2RGB)

h,w = image.shape[0:2]

rotationMatrix = cv2.getRotationMatrix2D((w/2, h/2), 200, .5)

rotated_image = cv2.warpAffine(image,rotationMatrix,(w,h))

c.imshow("Fist Lab",rotated_image)

c.waitKey(0)

c.destroyAllWindows()



![image](https://user-images.githubusercontent.com/72590669/104429161-4f1cfb00-55ab-11eb-9b71-6837db4e59ae.png)


# 3.Create a program to find sum and mean of a set of image.

In digital image processing, the sum of absolute differences (SAD) is a measure of the similarity between image blocks. It is calculated by taking the absolute difference between each pixel in the original block and the corresponding pixel in the block being used for comparison

To calculate the mean of all pixels in the image, without regard to what color channel they came from (if it's a color image), you do meanIntensity = mean
 import cv2
import os
path = 'C:\Pictures'
imgs = []

files = os.listdir(path)
for file in files:
    filepath=path+"\\"+file
    imgs.append(cv2.imread(filepath))
i=0
im = []
for im in imgs:
    #cv2.imshow(files[i],imgs[i])
    im+=imgs[i]
    i=i+1
cv2.imshow("sum of four pictures",im)
meanImg = im/len(files)
cv2.imshow("mean of four pictures",meanImg)
cv2.waitKey(0)

![image](https://user-images.githubusercontent.com/72590669/104430238-6d372b00-55ac-11eb-9982-b29f0cb0b818.png)
![image](https://user-images.githubusercontent.com/72590669/104430566-dae35700-55ac-11eb-9cb4-27644159716d.png)


# 4.Develop a program to convert image to binary image and gray scale.

Binary images are images whose pixels have only two possible intensity values. Numerically, the two values are often 0 for black, and either 1 or 255 for white. The main reason binary images are particularly useful in the field of Image Processing is because they allow easy separation of an object from the background.

In digital photography, computer-generated imagery, and colorimetry, a grayscale or image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.

 import cv2
img = cv2.imread('rose.jpg')
cv2.imshow('Input',img)
cv2.waitKey(0)
grayimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscaleimage',grayimg)
cv2.waitKey(0)
ret, bw_img = cv2.threshold(img,127,255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

![image](https://user-images.githubusercontent.com/72590669/104432424-06ffd780-55af-11eb-9fd4-4f2c9300df73.png)
![image](https://user-images.githubusercontent.com/72590669/104432631-37477600-55af-11eb-9b7a-833e077b481e.png)
![image](https://user-images.githubusercontent.com/72590669/104432856-71187c80-55af-11eb-95de-4b2ed6c4e76d.png)


# 5.Develop a program to convert given color image to different color space.
Color spaces are different types of color modes, used in image processing and signals and system for various purposes.
The color spaces in image processing aim to facilitate the specifications of colors in some standard way. Different types of color spaces are used in multiple fields like in hardware, in multiple applications of creating animation, etc.
import cv2
image=cv2.imread('rose.jpg')
cv2.imshow('pic',image)
cv2.waitKey(0)
yuv_img = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
cv2.imshow('ychannel',yuv_img[:,:,0])
cv2.imshow('uchannel',yuv_img[:,:,1])
cv2.imshow('vchannel',yuv_img[:,:,2])
cv2.waitKey(0)
hsv_img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
cv2.imshow('hchannel',hsv_img[:,:,0])
cv2.imshow('schannel',hsv_img[:,:,1])
cv2.imshow('vchannel',hsv_img[:,:,2])
cv2.waitKey(0)
cv2.destroyAllWindows()


![image](https://user-images.githubusercontent.com/72590669/104433843-8b068f00-55b0-11eb-92e5-726a4588ad5c.png)

# YUV

![image](https://user-images.githubusercontent.com/72590669/104434058-c903b300-55b0-11eb-94dd-be9ad43a32fc.png)
![image](https://user-images.githubusercontent.com/72590669/104434209-fcded880-55b0-11eb-9adb-39bb587a0bbc.png)
