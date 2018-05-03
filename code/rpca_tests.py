#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage import color
from skimage.transform import resize
from sklearn.utils.extmath import randomized_svd
from rpca import *


data = sio.loadmat('../data/demo_vid.mat')
M = data['M']
ht = np.asscalar(data['vh'])
wd = np.asscalar(data['vw'])

"""
data = sio.loadmat('../data/escalator_data.mat')
M = data['X'][:,:51]
ht = np.asscalar(data['m'])
wd = np.asscalar(data['n'])
"""

m,n = M.shape
dim = max(wd, ht)


rpca = RPCA(M)

# APG
s_apg = time.process_time()
rpca.rpca_apg()
e_apg = time.process_time()
L_apg = rpca.L_
S_apg = rpca.S_

# EALM
s_ealm = time.process_time()
rpca.rpca_ealm()
e_ealm = time.process_time()
L_ealm = rpca.L_
S_ealm = rpca.S_

# IALM
s_ialm = time.process_time()
rpca.rpca_ialm()
e_ialm = time.process_time()
L_ialm = rpca.L_
S_ialm = rpca.S_

print("Total time: " + str(e_apg - s_apg))
print("Total time: " + str(e_ealm - s_ealm))
print("Total time: " + str(e_ialm - s_ialm))


orig_esc  = resize(M[:,49].reshape(wd,ht).T, (dim,dim))
im_lr_apg = resize(L_apg[:,49].reshape(wd,ht).T,  (dim,dim))
im_sp_apg = resize(S_apg[:,49].reshape(wd,ht).T,  (dim,dim))
im_lr_ealm = resize(L_ealm[:,49].reshape(wd,ht).T,  (dim,dim))
im_sp_ealm = resize(S_ealm[:,49].reshape(wd,ht).T,  (dim,dim))
im_lr_ialm = resize(L_ialm[:,49].reshape(wd,ht).T,  (dim,dim))
im_sp_ialm = resize(S_ialm[:,49].reshape(wd,ht).T,  (dim,dim))

fig, ax = plt.subplots(3,3)
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

plt.show()
