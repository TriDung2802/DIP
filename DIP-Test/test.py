import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
# def conv(A,k):
#     kh,kw=k.shape
#     h,w=A.shape
#     B=np.ones((h,w))

#     for i in range(0,h-kh+1):
#         for j in range(0,w-kw+1):
#             sA=A[i:i+kh,j:j+kw]
#             B[i,j]=np.sum(sA*k)

#     return B


# def conv2(A,k,b=0):
#     kh,kw=k.shape

#     if b>0:
#         h,w=A.shape
#         B=np.ones((h+kh-1,w+kw-1))
#         th=int(kh/2)
#         tw=int(kw/2)
#         B[th:th+h,tw:tw+w]=A
#         A=B

#     h,w=A.shape
#     C=np.ones((h,w))

#     for i in range(0,h-kh+1):
#         for j in range(0,w-kw+1):
#             sA=A[i:i+kh,j:j+kw]
#             C[i,j]=np.sum(sA*k)

#     C=C[0:h-kh+1,0:w-kw+1]

#     return C


# img=cv2.imread("Images/meo.png",0)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# k=np.ones((5,5))/25

# r,g,b=cv2.split(img)

# B=conv(b,k)
# G=conv(g,k)
# R=conv(r,k)

# imgC=np.array(cv2.merge((R,G,B)),dtype='uint8')


# plt.figure()
# plt.imshow(img)
# plt.title("Anh goc")
# plt.savefig("anh_goc.png")

# plt.figure()
# plt.imshow(imgC)
# plt.title("Anh sau khi convolution")
# plt.savefig("anh_convolution.png")

# print("Da luu anh")

img =cv2.imread("Images/meo.png",0)
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
magnitude_spectrum=20*np.log(np.abs(fshift))
plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title('Input Image')
plt.xticks([]),plt.yticks([])
plt.subplot(122)
plt.imshow(magnitude_spectrum,cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([]),plt.yticks([])
plt.tight_layout()
plt.savefig("fft_result.png", dpi=150)
print("Da luu ket qua vao fft_result.png")