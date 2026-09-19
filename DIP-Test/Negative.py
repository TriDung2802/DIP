import cv2
# img = cv2.imread("Images/meo.png",cv2.IMREAD_COLOR)
# cv2.imshow("meo.png",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img =cv2.imread("Images/meo.png",cv2.IMREAD_COLOR)
img_neg=img.copy()
height,width=img.shape[:2]
for i in range(0,height):
    for j in range(0,width):
        img_neg[i,j]=255-img[i,j]
cv2.imshow("meo negative",img_neg)
cv2.waitKey(0)
cv2.destroyAllWindows()