from skimage.transform import *
from timeit import timeit
from rpca import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import time

def downscale(M, h, w, scale = 0.5):
	size = M.shape
	M0 = rescale(M[:, 0].reshape(w, h),scale)
	h_out = M0.shape[1]
	w_out = M0.shape[0]
	M_out = np.zeros((h_out * w_out, size[1]))
	for i in range(size[1]):
		Mi = rescale(M[:, i].reshape(w, h),scale)
		M_out[:, i] = Mi.reshape(-1, 1)[:, 0]
	return M_out, h_out, w_out


demo_data = sio.loadmat('../data/demo_vid.mat')
demo_M_1 = demo_data['M']
demo_h_1 = np.asscalar(demo_data['vh'])
demo_w_1 = np.asscalar(demo_data['vw'])


esca_data = sio.loadmat('../data/escalator_data.mat')
esca_M_1 = esca_data['X']
esca_h_1 = np.asscalar(esca_data['m'])
esca_w_1 = np.asscalar(esca_data['n'])


# esca_M_1, esca_h_1, esca_w_1 = downscale(esca_M_1, esca_h_1, esca_w_1, 0.1)

M = demo_M_1
h = demo_h_1
w = demo_w_1

rpca = RPCA(M)
rpca.rpca_ialm()

L = rpca.L_ 
S = rpca.S_

fig, axes = plt.subplots(1, 3)
axes[0].imshow(M[:, 4].reshape(w, h).T,	cmap = 'gray')
axes[1].imshow(L[:, 4].reshape(w, h).T, cmap = 'gray')
axes[2].imshow(S[:, 4].reshape(w, h).T, cmap = 'gray')
plt.show()
