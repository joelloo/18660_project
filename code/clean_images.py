import numpy as np
from skimage import io
from skimage import color
from rpca import *
from fastpcp import *
import matplotlib.pyplot as plt
import time

img = io.imread("../data/yalefaces/subject01.leftlight")
img = color.rgb2gray(img);
rpca = RPCA(img)
s_ialm = time.process_time()
rpca.rpca_ialm()
e_ialm = time.process_time()
L_ialm = rpca.L_
S_ialm = rpca.S_

s_ealm = time.process_time()
rpca.rpca_ealm()
e_ealm = time.process_time()
L_ealm = rpca.L_
S_ealm = rpca.S_

s_apg = time.process_time()
rpca.rpca_apg()
e_apg = time.process_time()
L_apg = rpca.L_
S_apg = rpca.S_

lm = 1 / np.sqrt(max(img.shape))
s_fpcp = time.process_time()
L_fpcp, S_fpcp = fastpcp(img, lm)
e_fpcp = time.process_time()


print("Total time: " + str(e_apg - s_apg))
print("Total time: " + str(e_ealm - s_ealm))
print("Total time: " + str(e_ialm - s_ialm))
print("Total time: " + str(e_fpcp - s_fpcp))
print("Size of image: " + str(img.shape))

orig_esc  = img 
im_lr_apg = L_apg
im_sp_apg = S_apg
im_lr_ealm = L_ealm
im_sp_ealm = S_ealm
im_lr_ialm = L_ialm
im_sp_ialm = S_ialm
im_lr_fpcp = L_fpcp
im_sp_fpcp = S_fpcp

fig, ax = plt.subplots(4,3)
fig.subplots_adjust(left=0.04, right=1, hspace=0.001, wspace=0)

ax[0,0].imshow(orig_esc, cmap='gray')
ax[0,0].set_title('Original')
ax[0,0].set_ylabel('APG')
ax[0,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

ax[0,1].imshow(im_lr_apg, cmap='gray')
ax[0,1].set_title('Low rank')
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)

ax[0,2].imshow(im_sp_apg, cmap='gray')
ax[0,2].set_title('Sparse')
ax[0,2].get_xaxis().set_visible(False)
ax[0,2].get_yaxis().set_visible(False)

ax[1,0].imshow(orig_esc, cmap='gray')
ax[1,0].set_ylabel('EALM')
ax[1,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

ax[1,1].imshow(im_lr_ealm, cmap='gray')
ax[1,1].get_xaxis().set_visible(False)
ax[1,1].get_yaxis().set_visible(False)

ax[1,2].imshow(im_sp_ealm, cmap='gray')
ax[1,2].get_xaxis().set_visible(False)
ax[1,2].get_yaxis().set_visible(False)

ax[2,0].imshow(orig_esc, cmap='gray')
ax[2,0].set_ylabel('IALM')
ax[2,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

ax[2,1].imshow(im_lr_ialm, cmap='gray')
ax[2,1].get_xaxis().set_visible(False)
ax[2,1].get_yaxis().set_visible(False)

ax[2,2].imshow(im_sp_ialm, cmap='gray')
ax[2,2].get_xaxis().set_visible(False)
ax[2,2].get_yaxis().set_visible(False)

ax[3,0].imshow(orig_esc, cmap='gray')
ax[3,0].set_ylabel('FPCP')
ax[3,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

ax[3,1].imshow(im_lr_fpcp, cmap='gray')
ax[3,1].get_xaxis().set_visible(False)
ax[3,1].get_yaxis().set_visible(False)

ax[3,2].imshow(im_sp_fpcp, cmap='gray')
ax[3,2].get_xaxis().set_visible(False)
ax[3,2].get_yaxis().set_visible(False)

plt.show()
