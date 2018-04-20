import numpy as np
from skimage import io
from rpca import *
import matplotlib.pyplot as plt

face_s1 = io.imread("/Users/joel/Documents/Spring 18/18660/project/data/yalefaces/subject01.leftlight")
rpca = RPCA(face_s1)
#rpca.rpca_ialm()
#rpca.rpca_ealm()
rpca.rpca_apg()

L_s1 = rpca.L_
S_s1 = rpca.S_

plt.figure()
plt.imshow(face_s1, cmap='gray')

plt.figure()
plt.imshow(L_s1, cmap='gray')

plt.figure()
plt.imshow(S_s1, cmap='gray')

plt.show()
