import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.utils.extmath import randomized_svd
from updateSVD import *


def incPCP_init(D_init, r):
    u, s, vt = randomized_svd(D_init, r)
    return u, s, vt


def incPCP_update(d, u0, s0, v0, lm, r, k, niters=2, bl=None):
    ui, si, vi = incrSVD(d, u0, s0, v0)
    vi = vi.T

    for j in range(niters):
        Lk = np.matmul(ui[:,:r], np.matmul(np.diag(si), np.atleast_2d(vi[-1,:]).T))
        Lk = Lk.flatten()
        Sk = softThresh(d - Lk, lm)
        if j == niters-1:
            break
        uk, sk, vk = replSVD(d - Sk, ui, si, vi)

    return Lk, Sk, uk, sk, vk


def softThresh(x, lm):
    return np.sign(x) * np.maximum(np.abs(x) - lm, 0)


def test():
    #data = sio.loadmat('../data/demo_vid.mat')
    #M = data['M']
    #ht = np.asscalar(data['vh'])
    #wd = np.asscalar(data['vw'])
    data = sio.loadmat('../data/escalator_data.mat')
    M = data['X']
    ht = np.asscalar(data['m'])
    wd = np.asscalar(data['n'])

    dim = max(wd, ht)
    lm = 1 / np.sqrt(max(M.shape))
    m,n = M.shape
    r = 2
    k0 = 3

    ui, si, vi = incPCP_init(M[:,:k0], r)
    uk = ui
    sk = si
    vk = vi.T

    for i in range(k0, n):
        d = M[:,i]
        print "d----"
        print d
        Lk, Sk, uk, sk, vk = incPCP_update(d, uk, sk, vk, lm, r, i)
        print "Lk----"
        print Lk
        print "Sk----"
        print Sk
        vk = vk.T

        if i%5 == 0:
            orig_demo = d.reshape(wd,ht).T
            im_lr_demo = Lk.reshape(wd,ht).T
            im_sp_demo = Sk.reshape(wd,ht).T

            fig, ax  = plt.subplots(1,3)
            fig.subplots_adjust(left=0.04, right=1, hspace=0.01, wspace=0)

            ax[0].imshow(orig_demo, cmap='gray')
            ax[0].set_title('Original')
            ax[0].set_ylabel('Highway')
            ax[0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            
            ax[1].imshow(im_lr_demo, cmap='gray')
            ax[1].set_title('Low rank')
            ax[1].get_xaxis().set_visible(False)
            ax[1].get_yaxis().set_visible(False)
            
            ax[2].imshow(im_sp_demo, cmap='gray')
            ax[2].set_title('Sparse')
            ax[2].get_xaxis().set_visible(False)
            ax[2].get_yaxis().set_visible(False)

            plt.show()



#test()


