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
from fralm import fralm
from ifralm import ifralm
from rpca import *

def decode(M, method = "default"):
    rpca = RPCA(M);
    if method == "default":
        rpca.rpca_cvx();
        L, S = rpca.L_, rpca.S_;
        return L,S
    elif method == "apg":
        rpca.rpca_apg();
        L, S = rpca.L_, rpca.S_;
        return L,S
    elif method == "ealm":
        rpca.rpca_ealm();
        L, S = rpca.L_, rpca.S_;
        return L,S
    elif method == "ialm":
        rpca.rpca_ialm();
        L, S = rpca.L_, rpca.S_;
        return L,S

def bitmap_to_mat(bitmap_seq):
    """from blog.shriphani.com"""
    matrix = []
    shape = None
    print("Number of frames:", len(bitmap_seq))
    for bitmap_file in bitmap_seq:
        # img = Image.open(bitmap_file).convert("L") %% rgb convert to gray
        img = Image.open(bitmap_file)
        if shape is None:
            shape = img.size
        assert img.size == shape
        img = np.array(img.getdata())
        matrix.append(img)
    return np.array(matrix), shape[::-1]


def do_plot(ax, img, shape):
    ax.cla()
    ax.imshow(img.reshape(shape), cmap="gray", interpolation="nearest")
    ax.set_xticklabels([])
    ax.set_yticklabels([])


if __name__ == "__main__":

    data = sio.loadmat('../data/demo_vid.mat')
    M = data['M']
    ht = np.asscalar(data['vh'])
    wd = np.asscalar(data['vw'])

    # data = sio.loadmat('../data/escalator_data.mat')
    # M = data['X'][:,:51]
    # ht = np.asscalar(data['m'])
    # wd = np.asscalar(data['n'])

    m,n = M.shape
    dim = max(wd, ht)

    start = time.time()
    # L, S = decode(M, method = "apg")
    L, S = fralm(M, 1, 0.003)
    time_lasting = time.time() - start;
    print("time of apg optimization is", time_lasting)
    orig_esc  = resize(M[:,49].reshape(wd,ht).T, (dim,dim))
    im_lr_esc = resize(L[:,49].reshape(wd,ht).T,  (dim,dim))
    im_sp_esc = resize(S[:,49].reshape(wd,ht).T,  (dim,dim))
    
    fig, ax = plt.subplots(1,3)
    fig.subplots_adjust(left=0.04, right=1, hspace=0.01, wspace=0)

    ax[0].imshow(orig_esc, cmap='gray')
    ax[0].set_title('Original Using APG')
    ax[0].set_ylabel('Face')
    ax[0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    ax[1].imshow(im_lr_esc, cmap='gray')
    ax[1].set_title('Low rank Using APG')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    ax[2].imshow(im_sp_esc, cmap='gray')
    ax[2].set_title('Sparse Using APG')
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False) 
    plt.show()

    # fig, axes = pl.subplots(1, 3, figsize=(10, 4))
    # fig.subplots_adjust(left=0, right=1, hspace=0, wspace=0.01)
    # for i in range(min(len(M), 500)):
    #     do_plot(axes[0], M[i], shape)
    #     axes[0].set_title("raw")
    #     do_plot(axes[1], L[i], shape)
    #     axes[1].set_title("low rank")
    #     do_plot(axes[2], S[i], shape)
    #     axes[2].set_title("sparse")
    #     fig.savefig("../data/results/traffic/{0:05d}.png".format(i))
