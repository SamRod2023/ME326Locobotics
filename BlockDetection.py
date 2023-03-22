import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def Center(color_img, color_mask):
    HSV_img = cv2.cvtColor( color_img, cv2.COLOR_RGB2HSV )
    if color_mask == "r": mask = cv2.inRange( HSV_img, (0, 100, 20), (10, 255, 255) ) + cv2.inRange( HSV_img, (160, 100, 20), (179, 255, 255) )
    elif color_mask == "g": mask = cv2.inRange( HSV_img, (55, 100, 20), (90, 255, 255) )
    elif color_mask == "b": mask = cv2.inRange( HSV_img, (97, 100, 20), (135, 255, 255) )
    elif color_mask == "y": mask = cv2.inRange( HSV_img, (15, 100, 20), (45, 255, 255) )
    dilation_img = cv2.dilate( mask, np.ones((6,6), np.uint8), iterations=1 ) 
    threshold = 600
    canny_img = cv2.Canny( dilation_img, threshold, threshold*2 )
    contours,_ = cv2.findContours( canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    X_bar = []; Y_bar = []; hull_list = []
    for cont in contours:
        if cv2.contourArea( cont ) > 500:
            hull = cv2.convexHull( cont ); hull_list.append( hull )
            M = cv2.moments( cont )
            cx = int(M["m10"] / (M["m00"]+0.001)); cy = int(M["m01"] / (M["m00"]+0.001))
            X_bar.append( cx ); Y_bar.append( cy )
    return X_bar, Y_bar

# Test

path = os.path.dirname(__file__)
# filename = os.path.join(path, "c1.jpg")
# filename = os.path.join(path, "c2.jpg")
# filename = os.path.join(path, "c3.jpg")
filename = os.path.join(path, "img.png")
c1_img = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
# Cx, Cy, mask, dilation, canny_img = Center(c1_img,color_mask='y')

plt.figure()
plt.subplot(221)
Cx, Cy = Center(c1_img,color_mask='r')
plt.imshow(c1_img)
plt.plot(Cx, Cy, 'w.', linewidth=0.5)
plt.xticks([]); plt.yticks([])
plt.subplot(222)
Cx, Cy = Center(c1_img,color_mask='b')
plt.imshow(c1_img)
plt.plot(Cx, Cy, 'w.', linewidth=0.5)
plt.xticks([]); plt.yticks([])
plt.subplot(223)
Cx, Cy = Center(c1_img,color_mask='y')
plt.imshow(c1_img)
plt.plot(Cx, Cy, 'w.', linewidth=0.5)
plt.xticks([]); plt.yticks([])
plt.subplot(224)
Cx, Cy = Center(c1_img,color_mask='g')
plt.imshow(c1_img)
plt.plot(Cx, Cy, 'w.', linewidth=0.5)
plt.xticks([]); plt.yticks([])
plt.show()