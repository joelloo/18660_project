import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd


def fastpcp(D, lm, tau=1e-2, n_iters=50):
    m,n = D.shape
    S = np.zeros((m,n))
    L = np.zeros((m,n))
    r = 1

    for k in range(n_iters):
        print "Iteration " + str(k) + " | Rank: " + str(r)
        u, s, vt = randomized_svd(D-S, n_components=r)
        L = np.matmul(u, np.matmul(np.diag(s), vt))
        if s[r-1] / np.sum(s) > tau:
            r += 1
        S = softThresh(D-L, lm)

    return L, S


def softThresh(x, lm):
    return np.sign(x) * np.maximum(np.abs(x) - lm, 0)


# Test
data = sio.loadmat('../data/demo_vid.mat')
M = data['M']
ht = np.asscalar(data['vh'])
wd = np.asscalar(data['vw'])

lm = 1 / np.sqrt(max(M.shape))
m,n = M.shape
L, S = fastpcp(M, lm, tau=0.05)

for i in range(0,n,5):
    im_lr = L[:,i].reshape(wd,ht).T
    im_sp = S[:,i].reshape(wd,ht).T
    #plt.imshow(im_lr, cmap='gray')
    plt.imshow(im_sp, cmap='gray')
    plt.show()
        

