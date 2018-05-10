#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage import color
from skimage.transform import resize
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import rand
from rpca import *
from fastpcp import *
from orpca import *
from ifastpcp import *
import time


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

lm1 = 1 / np.sqrt(max(M.shape))
lm2 = lm1

# iFPCP
r = 2
k0 = 3
lm = lm1
s = time.process_time()
uk, sk, vk = incPCP_init(M[:,:k0], r)
e = time.process_time()
vk = vk.T

fig, ax = plt.subplots(3,5)
fig.subplots_adjust(left=0.04, right=1, hspace=0.001, wspace=0)

lr = []
sp = []
orig = []
idxs = []
fpcp_err = []
timing = [e-s]

for i in range(k0, n):
    d = M[:,i]
    s = time.process_time()
    Lk, Sk, uk, sk, vk = incPCP_update(d, uk, sk, vk, lm, r, i)
    e = time.process_time()
    vk = vk.T

    fpcp_err.append(np.linalg.norm(M[:,i]-Lk-Sk))
    timing.append(e-s)

    if i%10 == 0:
        orig.append(M[:,i].reshape(wd,ht).T)
        lr.append(Lk.reshape(wd,ht).T)
        sp.append(Sk.reshape(wd,ht).T)
        idxs.append(i)

for idx in range(len(orig)):
    im_orig = orig[idx]
    im_lr = lr[idx]
    im_sp = sp[idx]

    if idx == 0:
        ax[0,0].imshow(im_orig, cmap='gray')
        ax[0,0].set_title('Frame ' + str(idxs[0]))
        ax[0,0].set_ylabel('Original')
        ax[0,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax[1,0].imshow(im_lr, cmap='gray')
        ax[1,0].set_ylabel('Low-rank')
        ax[1,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax[2,0].imshow(im_sp, cmap='gray')
        ax[2,0].set_ylabel('Sparse')
        ax[2,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    else:
        ax[0,idx].imshow(im_orig, cmap='gray')
        ax[0,idx].set_title('Frame ' + str(idxs[idx]))
        ax[0,idx].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax[1,idx].imshow(im_lr, cmap='gray')
        ax[1,idx].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax[2,idx].imshow(im_sp, cmap='gray')
        ax[2,idx].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

plt.show()


# ORPCA
L, S, stoc_err, am_err, stoc_timing = orpca(M, lm1, lm2, np.random.rand(m,n), "altmin", output_err=True)

fig, ax = plt.subplots(3,6)
fig.subplots_adjust(left=0.04, right=1, hspace=0.001, wspace=0)

idx = 0

for i in range(0, n, 10):
    im_orig = M[:,i].reshape(wd,ht).T
    im_lr = L[:,i].reshape(wd,ht).T
    im_sp = S[:,i].reshape(wd,ht).T

    if idx == 0:
        ax[0,0].imshow(im_orig, cmap='gray')
        ax[0,0].set_title('Frame 0')
        ax[0,0].set_ylabel('Original')
        ax[0,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax[1,0].imshow(im_lr, cmap='gray')
        ax[1,0].set_ylabel('Low-rank')
        ax[1,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


        ax[2,0].imshow(im_sp, cmap='gray')
        ax[2,0].set_ylabel('Sparse')
        ax[2,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    else:
        ax[0,idx].imshow(im_orig, cmap='gray')
        ax[0,idx].set_title('Frame ' + str(i))
        ax[0,idx].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax[1,idx].imshow(im_lr, cmap='gray')
        ax[1,idx].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax[2,idx].imshow(im_sp, cmap='gray')
        ax[2,idx].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    idx += 1

plt.show()

line1, = plt.plot(range(k0-1,n), timing)
line2, = plt.plot(range(len(stoc_timing)), stoc_timing)
plt.xlabel('Frame no.')
plt.ylabel('Time (s)')
plt.title('Runtime of update step for each frame')
plt.legend(handles=[line1,line2], labels=['Inc Fast PCP', 'Inc Stoc RPCA'])
plt.show()

line1, = plt.plot(range(0,n), stoc_err)
line2, = plt.plot(range(k0,n), fpcp_err)
plt.xlabel('Frame no.')
plt.ylabel('Error')
plt.title('Decomposition error as frames are added')
plt.legend(handles=[line1,line2], labels=['Inc Stoc RPCA', 'Inc FPCP'])
plt.show()

lines = []
for i in range(5):
    err = am_err[i]
    line, = plt.plot(range(len(err)), err)
    lines.append(line)

plt.xlabel('Inner loop iterations (alternating min)')
plt.ylabel('Error')
plt.title('Inner loop error over iterations for first 5 frames (Inc STOC-RPCA)')
plt.legend(handles=lines, labels=map(lambda x: 'Frame ' + str(x), range(len(lines))))
plt.show()

# Generate artificial data to test online algorithms on
init_vecs = np.random.rand(500,1) * 3
L = np.tile(init_vecs, 100)
m,n = L.shape
S = np.asarray(rand(m,n,density=0.01).todense()) * 100
M = L + S

r = 2
k0 = 3
lm = 1 / np.sqrt(max(M.shape))
uk, sk, vk = incPCP_init(M[:,:k0], r)
vk = vk.T

fpcp_err = []
lr_err = []
sp_err = []

for i in range(k0, n):
    d = M[:,i]
    Lk, Sk, uk, sk, vk = incPCP_update(d, uk, sk, vk, lm, r, i)
    vk = vk.T

    fpcp_err.append(np.linalg.norm(M[:,i]-Lk-Sk))
    lr_err.append(np.linalg.norm(Lk-L[:,i]))
    sp_err.append(np.linalg.norm(Sk-S[:,i]))

line1, = plt.plot(range(k0,n), lr_err, linestyle='--')
line2, = plt.plot(range(k0,n), sp_err)
#line3, = plt.plot(range(k0,n), fpcp_err)
#plt.legend(handles=[line1,line2,line3], labels=['Low-rank', 'Sparse', 'Convergence'])
plt.show()

exit(0)


