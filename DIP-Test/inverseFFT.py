import cv2 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

img=cv2.imread("Images/meo.png",0)
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
rows,cols=img.shape
crow,ccol=rows//2,cols//2
fshift[crow-30:crow+31,ccol-30:ccol+31]=0
f_ishift=np.fft.ifftshift(fshift)
img_back=np.fft.ifft2(f_ishift)
img_back=np.real(img_back)
plt.subplot(131)
plt.imshow(img,cmap='gray')
plt.title('Input Image')
plt.xticks([]),plt.yticks([])
plt.subplot(132)
plt.imshow(img_back,cmap='gray')
plt.title('Image after HPF')
plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(img_back,cmap='jet')
plt.title('Result in Jet')
plt.tight_layout()
plt.xticks([]),plt.yticks([])
plt.savefig('HPF.png',dpi=300)
