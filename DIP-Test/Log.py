import cv2
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

image_bgr =cv2.imread("Images/meo.png",cv2.IMREAD_COLOR)
image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image = np.array(image_bgr, 'float')

maxV = 255 / np.log(1 + np.max(image))
vals = np.linspace(0, maxV, 8, dtype=int)

plt.rcParams.update({'font.size': 7})
plt.figure(figsize=(3, 3), dpi=600)
fig = plt.figure(figsize=(3, 3), dpi=600)
subf = plt.subplot(3, 3, 1)
subf.imshow(image_bgr)
subf.set_title("Anh goc: ")
subf.axis('off')

for i, c in enumerate(vals):
    log_image = c * np.log(image + 1)
    log_image = np.array(log_image, dtype='uint8')
    subf = plt.subplot(3, 3, i + 2)
    subf.imshow(log_image)
    subf.set_title('c= ' + str(c))
    subf.axis('off')
fig.tight_layout()
fig.savefig("log_transform.png", dpi=300)
print("Đã lưu ảnh")
print(np.max(image_bgr))
print(maxV)
print(vals)