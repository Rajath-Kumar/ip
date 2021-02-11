# ip
# 1.develop a program to display grayscale image using read and write operation
In digital photography, computer-generated imagery, and colorimetry, a grayscale or image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.

To convert an image to grayscale in any of the Microsoft Office suite apps, right-click it and select Format Picture from the context menu . This will open an image editing panel on the right. Go to the Picture tab (the very last one). Expand the Picture Color options, and click the little dropdown next to the Presets for Color Saturation.

```
import cv2 as c
import numpy as np
image = c.imread("rose.jpg")
gray  = c.cvtColor(image,c.COLOR_BGR2GRAY)
c.imshow("Fist Lab",gray)
cv2.imwrite("rajath.jpg",grayimg)
c.waitKey(0)
c.destroyAllWindows()
```
# Output


![image](https://user-images.githubusercontent.com/72590669/104425684-f8adbd80-55a6-11eb-846c-90960482e44d.png)
![image](https://user-images.githubusercontent.com/72590669/104426836-7faf6580-55a8-11eb-9fc9-0ba7d7901ada.png)

# 2.Develop a program to perform linear transformation image scaling and rotation.
Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method. It is used for manipulation of an image so that the result is more suitable than the original for a specific application.

Image scaling is a computer graphics process that increases or decreases the size of a digital image. An image can be scaled explicitly with an image viewer or editing software, or it can be done automatically by a program to fit an image into a differently sized area.

Image rotation is a common image processing routine with applications in matching, alignment, and other image-based algorithms. The input to an image rotation routine is an image, the rotation angle θ, and a point about which rotation is done.

```
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
```
# Output
![image](https://user-images.githubusercontent.com/72590669/104428118-14669300-55aa-11eb-98ca-867d28539de4.png)



# Scaling
```
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
```
# Scaling 1
![image](https://user-images.githubusercontent.com/72590669/104428118-14669300-55aa-11eb-98ca-867d28539de4.png)

# scaling 2
![image](https://user-images.githubusercontent.com/72590669/104428437-7d4e0b00-55aa-11eb-876c-2ff0d8bf51a7.png)





# Rotation
```
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
```


![image](https://user-images.githubusercontent.com/72590669/104429161-4f1cfb00-55ab-11eb-9b71-6837db4e59ae.png)


# 3.Create a program to find sum and mean of a set of image.

In digital image processing, the sum of absolute differences (SAD) is a measure of the similarity between image blocks. It is calculated by taking the absolute difference between each pixel in the original block and the corresponding pixel in the block being used for comparison

Mean is most basic of all statistical measure. Means are often used in geometry and analysis; a wide range of means have been developed for these purposes. In contest of image processing filtering using mean is classified as spatial filtering and used for noise reduction.

```
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
```
# Output
![image](https://user-images.githubusercontent.com/72590669/104430238-6d372b00-55ac-11eb-9982-b29f0cb0b818.png)
![image](https://user-images.githubusercontent.com/72590669/104430566-dae35700-55ac-11eb-9cb4-27644159716d.png)


# 4.Develop a program to convert image to binary image and gray scale.

Binary images are images whose pixels have only two possible intensity values. Numerically, the two values are often 0 for black, and either 1 or 255 for white. The main reason binary images are particularly useful in the field of Image Processing is because they allow easy separation of an object from the background.

In digital photography, computer-generated imagery, and colorimetry, a grayscale or image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.
```
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
```
# Output
![image](https://user-images.githubusercontent.com/72590669/104432424-06ffd780-55af-11eb-9fd4-4f2c9300df73.png)
![image](https://user-images.githubusercontent.com/72590669/104432631-37477600-55af-11eb-9b7a-833e077b481e.png)
![image](https://user-images.githubusercontent.com/72590669/104432856-71187c80-55af-11eb-95de-4b2ed6c4e76d.png)


# 5.Develop a program to convert given color image to different color space.
Color spaces are different types of color modes, used in image processing and signals and system for various purposes.
The color spaces in image processing aim to facilitate the specifications of colors in some standard way. Different types of color spaces are used in multiple fields like in hardware, in multiple applications of creating animation, etc.

```
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
```
# Output
![image](https://user-images.githubusercontent.com/72590669/104433843-8b068f00-55b0-11eb-92e5-726a4588ad5c.png)

# YUV

![image](https://user-images.githubusercontent.com/72590669/104434058-c903b300-55b0-11eb-94dd-be9ad43a32fc.png)
![image](https://user-images.githubusercontent.com/72590669/104434209-fcded880-55b0-11eb-9adb-39bb587a0bbc.png)
![image](https://user-images.githubusercontent.com/72590669/104434373-2992f000-55b1-11eb-93d1-4bd28d574e1a.png)

# HSV

![image](https://user-images.githubusercontent.com/72590669/104434872-c5bcf700-55b1-11eb-8d57-7895e9dc171d.png)
![image](https://user-images.githubusercontent.com/72590669/104435005-ea18d380-55b1-11eb-947e-1eb7ec5e8328.png)
![image](https://user-images.githubusercontent.com/72590669/104435217-20eee980-55b2-11eb-8d3d-5d9aba6c141c.png)


# 6.DEVELOP A PROGRAM TO CREATE AN ARRAY FROM 2D ARRAY
For a two-dimensional array, in order to reference every element, we must use two nested loops. This gives us a counter variable for every column and every row in the matrix. int cols = 10; int rows = 10; int [] [] myArray = new int [cols] [rows]; // Two nested loops allow us to visit every spot in a 2D array
Creating Arrays. You can create an array by using the new operator with the following syntax − Syntax arrayRefVar = new dataType[arraySize]; The above statement does two things − It creates an array using new dataType[arraySize]. It assigns the reference of the newly created array to the variable arrayRefVar.
  
```  
  import numpy as np
from PIL import Image
import cv2

array = np.linspace(0,1,256*256)


mat = np.reshape(array,(256,256))

img = Image.fromarray(np.uint8(mat * 255) , 'L')
img.show()
cv2.waitKey(0)
array = np.linspace(0,1,256*256)


mat = np.reshape(array,(256,256))


img = Image.fromarray( mat , 'L')
img.show()
cv2.waitKey(0)
```
# Output
![image](https://user-images.githubusercontent.com/72590669/104436024-1aad3d00-55b3-11eb-9ae8-abbe47102946.png)
![image](https://user-images.githubusercontent.com/72590669/104436233-5d6f1500-55b3-11eb-8238-e17dec143dff.png)

# 7.Find the neighborhood matrix.
In topology and related areas of mathematics, a neighbourhood (or neighborhood) is one of the basic concepts in a topological space.It is closely related to the concepts of open set and interior.Intuitively speaking, a neighbourhood of a point is a set of points containing that point where one can move some amount in any direction away from that point without leaving the set.

import numpy as np
```
axis = 3
x =np.empty((axis,axis))
y = np.empty((axis+2,axis+2))
s =np.empty((axis,axis))
x = np.array([[1,4,3],[2,8,5],[3,4,6]])


'''
for i in range(0,axis):
    for j in range(0,axis):
        print(int(x[i][j]),end = '\t')
    print('\n')'''

print('Temp matrix\n')

for i in range(0,axis+2):
    for j in range(0,axis+2):
        if i == 0 or i == axis+1 or j == 0 or j==axis+1:
            y[i][j]=0
        else:
            #print("i = {}, J = {}".format(i,j))
            y[i][j]=x[i-1][j-1]
           

for i in range(0,axis+2):
    for j in range(0,axis+2):
        print(int(y[i][j]),end = '\t')
    print('\n')
   
   
print('Output calculated Neigbhors of matrix\n')      
for i in range(0,axis):
    for j in range(0,axis):
        s[i][j]=((y[i][j]+y[i][j+1]+y[i][j+2]+y[i+1][j]+y[i+1][j+2]+y[i+2][j]+y[i+2][j+1]+y[i+2][j+2])/8)
        print(s[i][j],end = '\t')
    print('\n')
```


   Output

![image](https://user-images.githubusercontent.com/72590669/104447005-34ee1780-55c1-11eb-9a62-7c72c960cc06.png)



# 8)To find the sum of neighbor matrix  
  
  Given a M x N matrix, find sum of all K x K sub-matrix 2. Given a M x N matrix and a cell (i, j), find sum of all elements of the matrix in constant time except the elements present at row i & column j of the matrix. Given a M x N matrix, calculate maximum sum submatrix of size k x k in a given M x N matrix in O (M*N) time. Here, 0 < k < M, N.
  ```
import numpy as np

axis = 3
x =np.empty((axis,axis))
y = np.empty((axis+2,axis+2))
r=np.empty((axis,axis))
s =np.empty((axis,axis))
x = np.array([[1,4,3],[2,8,5],[3,4,6]])


print('Matrix\n')

for i in range(0,axis):
    for j in range(0,axis):
        print(int(x[i][j]),end = '\t')
    print('\n')

print('Temp matrix\n')

for i in range(0,axis+2):
    for j in range(0,axis+2):
        if i == 0 or i == axis+1 or j == 0 or j==axis+1:
            y[i][j]=0
        else:
            #print("i = {}, J = {}".format(i,j))
            y[i][j]=x[i-1][j-1]
           

for i in range(0,axis+2):
    for j in range(0,axis+2):
        print(int(y[i][j]),end = '\t')
    print('\n')
   
   
print('Output calculated Neighbours of matrix\n')


 
print('sum of Neighbours of matrix\n')
for i in range(0,axis):
    for j in range(0,axis):
       
       
       
   r[i][j]=((y[i][j]+y[i][j+1]+y[i][j+2]+y[i+1][j]+y[i+1][j+2]+y[i+2][j]+y[i+2][j+1]+y[i+2][j+2]))
   print(r[i][j],end = '\t')
       
 print('\n')

print('\n Average of Neighbours of matrix\n')

for i in range(0,axis):
    for j in range(0,axis):
       
       
   s[i][j]=((y[i][j]+y[i][j+1]+y[i][j+2]+y[i+1][j]+y[i+1][j+2]+y[i+2][j]+y[i+2][j+1]+y[i+2][j+2])/8)
       
   print(s[i][j],end = '\t')
  print('\n')
```
 OUTPUT
 
 ![image](https://user-images.githubusercontent.com/72590669/104446506-92359900-55c0-11eb-9e43-39335b35a0ff.png)
![image](https://user-images.githubusercontent.com/72590669/104446674-ce68f980-55c0-11eb-90b3-fdf6887b435c.png)

#9) Wwrite a program to implement negative transformation

When an image is inverted, each of its pixel value 'r' is subtracted from the maximum pixel value L-1 and the original pixel is replaced with the result 's'. Image inversion or Image negation helps finding the details from the darker regions of the image.
Image is also known as a set of pixels. When we store an image in computers or digitally, it’s corresponding pixel values are stored. So, when we read an image to a variable using OpenCV in Python, the variable stores the pixel values of the image. When we try to negatively transform an image, the brightest areas are transformed into the darkest and the darkest areas are transformed into the brightest.
```
import cv2
import numpy as np
img=cv2.imread('rose.jpg')
cv2.imshow('original',img)
cv2.waitKey(0)
img_neg=255-img
cv2.imshow('negative',img_neg)
cv2.waitKey(0)
```
#OUTPUT

![image](https://user-images.githubusercontent.com/72590669/105326531-da257300-5bf3-11eb-829a-5e290d706d24.png)
![image](https://user-images.githubusercontent.com/72590669/105326903-47d19f00-5bf4-11eb-82b2-9a482b3e59d1.png)

#10) develop a program to implement contrast enhancement and brightness tresholding

#- contrast enhancement
Contrast refers to the amount of differentiation that is there between the various image features. Images having a higher contrast level generally display a greater degree of color or gray-scale variation than those of lower contrast.
Contrast Enhancement refers to the sharpening of image features to remove the noisy feature such as edges and contrast boundaries. Contrast Enhancement Algorithms aim to improve the perception of the image by human eye.

```
from PIL import Image, ImageEnhance
img = Image.open("rose.jpg")
img.show()
img=ImageEnhance.Color(img)
img.enhance(2.0).show()
```

![image](https://user-images.githubusercontent.com/72590669/105328710-58831480-5bf6-11eb-9a5b-377628f7ee42.png)
![image](https://user-images.githubusercontent.com/72590669/105328887-88321c80-5bf6-11eb-97cc-2b2fa1cc8af3.png)


#- brightness tresholding

Here, the matter is straight-forward. For every pixel, the same threshold value is applied. If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value. The function cv.threshold is used to apply the thresholding. The first argument is the source image, which should be a grayscale image. The second argument is the threshold value which is used to classify the pixel values. The third argument is the maximum value which is assigned to pixel values exceeding the threshold. OpenCV provides different types of thresholding which is given by the fourth parameter of the function. Basic thresholding as described above is done by using the type cv.THRESH_BINARY. All simple thresholding types are:

cv.THRESH_BINARY
cv.THRESH_BINARY_INV
cv.THRESH_TRUNC
cv.THRESH_TOZERO
cv.THRESH_TOZERO_INV

```
import cv2  
import numpy as np  
 

image1 = cv2.imread('rose.jpg')  
 
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
 

ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
 

cv2.imshow('Binary Threshold', thresh1)
cv2.imshow('Binary Threshold Inverted', thresh2)
cv2.imshow('Truncated Threshold', thresh3)
cv2.imshow('Set to 0', thresh4)
cv2.imshow('Set to 0 Inverted', thresh5)
 
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  
```    
    
    
# Output
![image](https://user-images.githubusercontent.com/72590669/105329541-45bd0f80-5bf7-11eb-9ccb-5011ef20ce02.png)

![image](https://user-images.githubusercontent.com/72590669/105329739-7dc45280-5bf7-11eb-9052-e9be5241dd3f.png)


![image](https://user-images.githubusercontent.com/72590669/105330219-0f33c480-5bf8-11eb-8814-f73c266f9164.png)

![image](https://user-images.githubusercontent.com/72590669/105330485-59b54100-5bf8-11eb-8c88-f17c3091e863.png)

![image](https://user-images.githubusercontent.com/72590669/105330633-88cbb280-5bf8-11eb-9794-b1b42b6a9cac.png)


# 11)develop a program toimplement power law (gamma) transformation

A variety of devices for image capture, printing, and display respond according to a power law. The exponent in power law equation is referred to as gamma Þ process used to correct this power law response phenomena is called gamma correction. ... With =2.5, the CRT would produce images darker than intended.
eg. i/p is a simple gray scale linear wedge .To reproduce colors accurately also requires knowledge of gamma correction as changing the value of gamma changes not only the brightness but also the color ratios of R:G:B.

```
import numpy as np
import cv2

img = cv2.imread('rose.jpg')

gamma_two_point_two = np.array(230*(img/255)**2.1,dtype='uint8')
# Similarly, Apply Gamma=0.4
gamma_point_four = np.array(255*(img/255)**0.1,dtype='uint8')
# Display the images in subplots
img3 = cv2.hconcat([gamma_two_point_two,gamma_point_four])
cv2.imshow('a2',img3)
cv2.waitKey(0)
```

#OUTPUT

![image](https://user-images.githubusercontent.com/72590669/105332040-2e335600-5bfa-11eb-9fc0-593600fc38e5.png)


# 12) program to enhanhce image using image artghmetic and logic operations.
  Arithmetic Operations like Addition, Subtraction, and Bitwise Operations(AND, OR, NOT, XOR) can be applied to the input images. These operations can be helpful in enhancing the properties of the input images. The Image arithmetics are important for analyzing the input image properties. The operated images can be further used as an enhanced input image, and many more operations can be applied for clarifying, thresholding, dilating etc of the image  
 image arithmetic applies one of the standard arithmetic operations or a logical operator to two or more images. The operators are applied in a pixel-by-pixel way, i.e. the value of a pixel in the output image depends only on the values of the corresponding pixels in the input images. Hence, the images must be of the same size. Although image arithmetic is the most simple form of image processing, there is a wide range of applications. A main advantage of arithmetic operators is that the process is very simple and therefore fast.Logical operators are often used to combine two (mostly binary) images. In the case of integer images, the logical operator is normally applied in a bitwise way.

  ```
 # addition of 2 image

      import cv2  
import numpy as np  
image1 = cv2.imread('mypet1.jpg')  
image2 = cv2.imread('mypet2.jpg')
weightedSum = cv2.addWeighted(image1, 0.5, image2, 0.4, 0)
cv2.imshow('Weighted Image', weightedSum)
if cv2.waitKey(0) & 0xff == 25:  
    cv2.destroyAllWindows()  
```
# Output

![image](https://user-images.githubusercontent.com/72590669/107619465-36bb0180-6c79-11eb-9062-3b2f4dc2db6e.png)


 # Subtraction of 2 images
 
 ```
import cv2  
import numpy as np  
image1 = cv2.imread('mypet1.jpg')  
image2 = cv2.imread('mypet2.jpg')
sub = cv2.subtract(image1, image2)
cv2.imshow('Subtracted Image', sub)
if cv2.waitKey(0) & 0xff == 27:  
  cv2.destroyAllWindows() 
```
![image](https://user-images.githubusercontent.com/72590669/107619876-d5476280-6c79-11eb-9800-ee5309d9bef5.png)

# Logical Operators
```
 import cv2
img1 = cv2.imread("mypet1.jpg")
img2 = cv2.imread("mypet2.jpg")
bitwise_and = cv2.bitwise_and(img2, img1)
cv2.imshow("bit_and", bitwise_and)
bitwise_or = cv2.bitwise_or(img2, img1)
cv2.imshow("bitwise_or", bitwise_or)
bitwise_xor = cv2.bitwise_xor(img2, img1)
cv2.imshow("bitwise_xor", bitwise_xor)
bitwise_not = cv2.bitwise_not(img1)
cv2.imshow("bitwise_not1", bitwise_not)
bitwise_not = cv2.bitwise_not(img2)
cv2.imshow("bitwise_not2", bitwise_not)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# Output
![image](https://user-images.githubusercontent.com/72590669/107620537-da58e180-6c7a-11eb-8b39-97aaa8ef665b.png)


![image](https://user-images.githubusercontent.com/72590669/107620637-05dbcc00-6c7b-11eb-8230-ff9cf609d66c.png)

![image](https://user-images.githubusercontent.com/72590669/107620761-391e5b00-6c7b-11eb-8d17-58589be51db4.png)

![image](https://user-images.githubusercontent.com/72590669/107620856-62d78200-6c7b-11eb-8fba-4354bf392248.png)

![image](https://user-images.githubusercontent.com/72590669/107621166-d8dbe900-6c7b-11eb-9ebe-bc595236e45c.png)

13) program for gray level slicing with and without background.

two different approaches are adopted for gray level slicing are
> gray level slicing with background
>grayb level slicing without background

gray level slicing with background it displays high level values in specific region of an image andgray level to other region by peserving background
gray level slicing without background  it displays high level values in specific region of an image andgray level to other region by ignoring background

# gray level slicing with background 
```
import cv2
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread('app.jpg',0)
x,y=image.shape
z=np.zeros((x,y))
for i in range(0,x):
    for j in range(0,y):
        if(image[i][j]>50 and image[i][j]<150):
            z[i][j]=255
        else:
            z[i][j]=image[i][j]
equ=np.hstack((image,z))
plt.title('Original ||   Graylevel slicing with background')
plt.imshow(equ,'gray')
plt.show()
```

# Output
 
![image](https://user-images.githubusercontent.com/72590669/107622773-51dc4000-6c7e-11eb-9bbf-13cbb6ec0031.png)
   
   
 # Gray level slicing without background
 ```
  import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image=cv2.imread('app.jpg',0)
x,y=image.shape
z=np.zeros((x,y))
for i in range(0,x):
    for j in range(0,y):
        if(image[i][j]>50 and image[i][j]<150):
            z[i][j]=255
        else:
            z[i][j]=0
equ=np.hstack((image,z))
plt.title('Original ||  Graylevel slicing w/o background')
plt.imshow(equ,'gray')
plt.show()
```

#Output


![image](https://user-images.githubusercontent.com/72590669/107623210-00808080-6c7f-11eb-9d75-74a17b3c4eb6.png)
  
  
 # 14) program for image enhancement using histogram equalization.
 
 Histogram equalization is a method in image processing of contrast adjustment using the image’s histogram.

This method usually increases the global contrast of many images, especially when the usable data of the image is represented by close contrast values. Through this adjustment, the intensities can be better distributed on the histogram. This allows for areas of lower local contrast to gain a higher contrast. Histogram equalization accomplishes this by effectively spreading out the most frequent intensity values. The method is useful in images with backgrounds and foregrounds that are both bright or both dark.

OpenCV has a function to do this, cv2.equalizeHist(). Its input is just grayscale image and output is our histogram equalized image.

```
from matplotlib.pyplot import imread, imshow, show, subplot, title, get_cmap, hist
from skimage.exposure import equalize_hist
import numpy as np


img = imread('messi.jpg')
eq = np.asarray(equalize_hist(img) * 255, dtype='uint8')

subplot(221); imshow(img, cmap=get_cmap('gray')); title('Original')
subplot(222); hist(img.flatten(), 256, range=(0,256)); title('Histogram of origianl')
subplot(223); imshow(eq, cmap=get_cmap('gray'));  title('Histogram Equalized')
subplot(224); hist(eq.flatten(), 256, range=(0,256));

show()
```

# Output


