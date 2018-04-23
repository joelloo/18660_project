#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from rpca import *
import iFrALM
import scipy.io as sio



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
    elif method == "iFrALM":
        L, S = iFrALM(M, fixed_rank = 1);
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
    import sys
    import glob
    import matplotlib.pyplot as pl

    if "--test" in sys.argv:
        M = (10*np.ones((10, 10))) + (-5 * np.eye(10))
        L, S, svd = pcp(M, verbose=True, svd_method="exact")
        assert np.allclose(M, L + S), "Failed"
        print("passed")
        sys.exit(0)


    alg = "all"
    # M, shape = bitmap_to_mat(glob.glob("../data/frames/traffic_downsampled/*.jpg")[:2000:1])
    data = sio.loadmat('../data/demo_vid.mat')
    M = data['M']
    ht = np.asscalar(data['vh'])
    wd = np.asscalar(data['vw'])

    lm = 1 / np.sqrt(max(M.shape))
    m,n = M.shape
    if alg == 'all' :

        start = time.time()
        L1, S1 = decode(M, method = "apg")
        time_lasting1 = time.time() - start;
        print("time of apg optimization is", time_lasting1)

        start = time.time()
        L2, S2 = decode(M, method = "ealm")
        time_lasting2 = time.time() - start;
        print("time of ealm optimization is", time_lasting2)

        start = time.time()
        L3, S3 = decode(M, method = "ialm")
        time_lasting3 = time.time() - start;
        print("time of ialm optimization is", time_lasting3)

    elif alg == 'default' :
        start = time.time()
        L, S = decode(M)
        time_lasting = time.time() - start;
        print("time of cvx optimization is", time_lasting)
    
    elif alg == 'apg' :
        start = time.time()
        L, S = decode(M, method = "apg")
        time_lasting = time.time() - start;
        print("time of apg optimization is", time_lasting)

    elif alg == 'ealm' :
        start = time.time()
        L, S = decode(M, method = "ealm")
        time_lasting = time.time() - start;
        print("time of ealm optimization is", time_lasting)

    elif alg == 'ialm' :
        start = time.time()
        L, S = decode(M, method = "ialm")
        time_lasting = time.time() - start;
        print("time of ialm optimization is", time_lasting)

    elif alg == 'iFrALM' :
        start = time.time()
        L, S = decode(M, method = "iFrALM")
        time_lasting = time.time() - start;
        print("time of iFrALM optimization is", time_lasting)

    if alg == 'all' :
        orig_demo = M[:,49].reshape(wd,ht).T
        im_lr_demo1 = L1[:,49].reshape(wd,ht).T
        im_sp_demo1 = S1[:,49].reshape(wd,ht).T
        im_lr_demo2 = L2[:,49].reshape(wd,ht).T
        im_sp_demo2 = S2[:,49].reshape(wd,ht).T
        im_lr_demo3 = L3[:,49].reshape(wd,ht).T
        im_sp_demo3 = S3[:,49].reshape(wd,ht).T

        fig, ax = plt.subplots(3,3)

        ax[0, 0].imshow(orig_demo, cmap='gray')
        ax[0, 0].set_title('Original Using APG')
        ax[0, 0].set_ylabel('Highway')
        ax[0, 0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax[0, 1].imshow(im_lr_demo1, cmap='gray')
        ax[0, 1].set_title('Low rank Using APG')
        ax[0, 1].get_xaxis().set_visible(False)
        ax[0, 1].get_yaxis().set_visible(False)

        ax[0, 2].imshow(im_sp_demo1, cmap='gray')
        ax[0, 2].set_title('Sparse Using APG')
        ax[0, 2].get_xaxis().set_visible(False)
        ax[0, 2].get_yaxis().set_visible(False) 

        ax[1, 0].imshow(orig_demo, cmap='gray')
        ax[1, 0].set_title('Original Using EALM')
        ax[1, 0].set_ylabel('Highway')
        ax[1, 0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax[1, 1].imshow(im_lr_demo2, cmap='gray')
        ax[1, 1].set_title('Low rank Using EALM')
        ax[1, 1].get_xaxis().set_visible(False)
        ax[1, 1].get_yaxis().set_visible(False)

        ax[1, 2].imshow(im_sp_demo2, cmap='gray')
        ax[1, 2].set_title('Sparse Using EALM')
        ax[1, 2].get_xaxis().set_visible(False)
        ax[1, 2].get_yaxis().set_visible(False)

        ax[2, 0].imshow(orig_demo, cmap='gray')
        ax[2, 0].set_title('Original Using IALM')
        ax[2, 0].set_ylabel('Highway')
        ax[2, 0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax[2, 1].imshow(im_lr_demo3, cmap='gray')
        ax[2, 1].set_title('Low rank Using IALM')
        ax[2, 1].get_xaxis().set_visible(False)
        ax[2, 1].get_yaxis().set_visible(False)

        ax[2, 2].imshow(im_sp_demo3, cmap='gray')
        ax[2, 2].set_title('Sparse Using IALM')
        ax[2, 2].get_xaxis().set_visible(False)
        ax[2, 2].get_yaxis().set_visible(False)

        plt.show()
        # plt.figure()
        # plt.plot




    # orig_demo = M[:,49].reshape(wd,ht).T
    # im_lr_demo = L[:,49].reshape(wd,ht).T
    # im_sp_demo = S[:,49].reshape(wd,ht).T


    # fig, ax = plt.subplots(1,3)
    # fig.subplots_adjust(left=0.04, right=1, hspace=0.01, wspace=0)

    # ax[0].imshow(orig_demo, cmap='gray')
    # ax[0].set_title('Original')
    # ax[0].set_ylabel('Highway')
    # ax[0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # ax[1].imshow(im_lr_demo, cmap='gray')
    # ax[1].set_title('Low rank')
    # ax[1].get_xaxis().set_visible(False)
    # ax[1].get_yaxis().set_visible(False)

    # ax[2].imshow(im_sp_demo, cmap='gray')
    # ax[2].set_title('Sparse')
    # ax[2].get_xaxis().set_visible(False)
    # ax[2].get_yaxis().set_visible(False)

    # plt.show()

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
