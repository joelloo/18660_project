import numpy as np
from skimage import io
from skimage import color
from rpca import *
import matplotlib.pyplot as plt

# face_s1 = io.imread("../data/yalefaces/subject01.leftlight")
# img = io.imread("../data/test/shot.jpg")
img = io.imread("../data/yalefaces/subject01.leftlight")
img = color.rgb2gray(img);
print img.shape
rpca = RPCA(img)
# rpca.rpca_ialm()
rpca.rpca_ealm()
# rpca.rpca_apg()

L_s1 = rpca.L_
S_s1 = rpca.S_

plt.figure()
plt.imshow(img, cmap='gray')

plt.figure()
plt.imshow(L_s1, cmap='gray')

plt.figure()
plt.imshow(S_s1, cmap='gray')

plt.show()
