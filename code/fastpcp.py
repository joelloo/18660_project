import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from skimage.transform import resize


def fastpcp(D, lm, tau=1e-2, n_iters=50):
    m,n = D.shape
    S = np.zeros((m,n))
    L = np.zeros((m,n))
    r = 1

    for k in range(n_iters):
        #print "Iteration " + str(k) + " | Rank: " + str(r)
        u, s, vt = randomized_svd(D-S, n_components=r)
        L = np.matmul(u, np.matmul(np.diag(s), vt))
        if s[r-1] / np.sum(s) > tau:
            r += 1
        S = softThresh(D-L, lm)

    print("Final error: " + str(np.linalg.norm(D-L-S, 'fro')) + " | " + str(r))
    return L, S


def softThresh(x, lm):
    return np.sign(x) * np.maximum(np.abs(x) - lm, 0)

"""
# Test on highway data
data = sio.loadmat('../data/demo_vid.mat')
M = data['M']
ht = np.asscalar(data['vh'])
wd = np.asscalar(data['vw'])

lm = 1 / np.sqrt(max(M.shape))
m,n = M.shape
L, S = fastpcp(M, lm, tau=0.05)

orig_demo = M[:,49].reshape(wd,ht).T
im_lr_demo = L[:,49].reshape(wd,ht).T
im_sp_demo = S[:,49].reshape(wd,ht).T

# Test on escalator data
data = sio.loadmat('../data/escalator_data.mat')
M = data['X'][:,:51]
ht = np.asscalar(data['m'])
wd = np.asscalar(data['n'])

lm = 1 / np.sqrt(max(M.shape))
m,n = M.shape
L, S = fastpcp(M, lm, tau=0.05)

dim = max(wd,ht)
orig_esc = resize(M[:,49].reshape(wd,ht).T, (dim,dim))
im_lr_esc = resize(L[:,49].reshape(wd,ht).T, (dim,dim))
im_sp_esc = resize(S[:,49].reshape(wd,ht).T, (dim,dim))

# Test on traffic data
data = sio.loadmat('../data/escalator_data.mat')
#M = data['X'][:,:51]
M = data['X']
ht = np.asscalar(data['m'])
wd = np.asscalar(data['n'])

lm = 1 / np.sqrt(max(M.shape))
m,n = M.shape
L, S = fastpcp(M, lm, tau=0.05)

dim = max(wd,ht)
orig_esc = resize(M[:,49].reshape(wd,ht).T, (dim,dim))
im_lr_esc = resize(L[:,49].reshape(wd,ht).T, (dim,dim))
im_sp_esc = resize(S[:,49].reshape(wd,ht).T, (dim,dim))

# Plot
fig, ax = plt.subplots(2,3)
fig.subplots_adjust(left=0.04, right=1, hspace=0.01, wspace=0)

ax[0,0].imshow(orig_demo, cmap='gray')
ax[0,0].set_title('Original')
ax[0,0].set_ylabel('Highway')
ax[0,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

ax[0,1].imshow(im_lr_demo, cmap='gray')
ax[0,1].set_title('Low rank')
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)

ax[0,2].imshow(im_sp_demo, cmap='gray')
ax[0,2].set_title('Sparse')
ax[0,2].get_xaxis().set_visible(False)
ax[0,2].get_yaxis().set_visible(False)

ax[1,0].imshow(orig_esc, cmap='gray')
ax[1,0].set_ylabel('Escalator')
ax[1,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

ax[1,1].imshow(im_lr_esc, cmap='gray')
ax[1,1].get_xaxis().set_visible(False)
ax[1,1].get_yaxis().set_visible(False)

ax[1,2].imshow(im_sp_esc, cmap='gray')
ax[1,2].get_xaxis().set_visible(False)
ax[1,2].get_yaxis().set_visible(False)

plt.show()
"""
