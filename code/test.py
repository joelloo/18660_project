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

M = esca_M_1
h = esca_h_1
w = esca_w_1

# M, h, w = downscale(M, h, w, 0.5)

rpca = RPCA(M)

rpca.rpca_ialm_v1()
L_ialm = rpca.L_ 
S_ialm = rpca.S_

# rpca.rpca_apg()
# L_apg = rpca.L_ 
# S_apg = rpca.S_

# rpca.rpca_ealm()
# L_ealm = rpca.L_ 
# S_ealm = rpca.S_

fig, axes = plt.subplots(1, 3)
axes[0].imshow(M[:, 4].reshape(w, h).T, cmap = 'gray')
axes[1].imshow(L_ialm[:, 4].reshape(w, h).T, cmap = 'gray')
axes[2].imshow(S_ialm[:, 4].reshape(w, h).T, cmap = 'gray')

'''
fig, axes = plt.subplots(3, 3)
fig.subplots_adjust(left=0.04, right=1, hspace=0.01, wspace=0)

axes[0, 0].imshow(M[:, 4].reshape(w, h).T, 		cmap = 'gray')
axes[0, 0].set_title('Original matrix')
axes[0, 0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

axes[0, 1].imshow(L_apg[:, 4].reshape(w, h).T, 	cmap = 'gray')
axes[0, 1].set_title('Low rank matrix')
axes[0, 1].get_xaxis().set_visible(False)
axes[0, 1].get_yaxis().set_visible(False)

axes[0, 2].imshow(S_apg[:, 4].reshape(w, h).T, 	cmap = 'gray')
axes[0, 2].set_title('Sparse matrix')
axes[0, 2].get_xaxis().set_visible(False)
axes[0, 2].get_yaxis().set_visible(False)

axes[1, 0].imshow(M[:, 4].reshape(w, h).T, 		cmap = 'gray')
# axes[1, 0].set_title('Original matrix')
axes[1, 0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

axes[1, 1].imshow(L_ealm[:, 4].reshape(w, h).T, cmap = 'gray')
# axes[1, 1].set_title('EALM: Low rank matrix')
axes[1, 1].get_xaxis().set_visible(False)
axes[1, 1].get_yaxis().set_visible(False)

axes[1, 2].imshow(S_ealm[:, 4].reshape(w, h).T, cmap = 'gray')
# axes[1, 2].set_title('EALM: Sparse matrix')
axes[1, 2].get_xaxis().set_visible(False)
axes[1, 2].get_yaxis().set_visible(False)

axes[2, 0].imshow(M[:, 4].reshape(w, h).T, 		cmap = 'gray')
# axes[2, 0].set_title('Original matrix')
axes[2, 0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

axes[2, 1].imshow(L_ialm[:, 4].reshape(w, h).T, cmap = 'gray')
# axes[2, 1].set_title('IALM: Low rank matrix')
axes[2, 1].get_xaxis().set_visible(False)
axes[2, 1].get_yaxis().set_visible(False)

axes[2, 2].imshow(S_ialm[:, 4].reshape(w, h).T, cmap = 'gray')
# axes[2, 2].set_title('IALM: Sparse matrix using')
axes[2, 1].get_xaxis().set_visible(False)
axes[2, 1].get_yaxis().set_visible(False)
'''

plt.show()
