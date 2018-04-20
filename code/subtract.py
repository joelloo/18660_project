#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from rpca import *



def decode(M, method = "default"):
    rpca = RPCA(M);
    if method == "default":
        rpca.rpca_cvx();
        L, S = rpca.L, rpca.S;
        return L,S
    elif method == "apg":
        rpca.rpca_apg();
        L, S = rpca.L, rpca.S;
        return L,S
    elif method == "ealm":
        rpca.rpca_ealm();
        L, S = rpca.L, rpca.S;
        return L,S
    elif method == "ialm":
        rpca.rpca_ialm();
        L, S = rpca.L, rpca.S;
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

    M, shape = bitmap_to_mat(glob.glob("../data/frames/traffic/*.jpg")[:2000:10])
    start = time.time()
    L, S = decode(M, method = "ealm")
    time_lasting = time.time() - start;
    print("time of cvx optimization is", time_lasting)
    # L, S, (u, s, v) = pcp(M, maxiter=100, verbose=True, svd_method="exact")

    fig, axes = pl.subplots(1, 3, figsize=(10, 4))
    fig.subplots_adjust(left=0, right=1, hspace=0, wspace=0.01)
    for i in range(min(len(M), 500)):
        do_plot(axes[0], M[i], shape)
        axes[0].set_title("raw")
        do_plot(axes[1], L[i], shape)
        axes[1].set_title("low rank")
        do_plot(axes[2], S[i], shape)
        axes[2].set_title("sparse")
        fig.savefig("../data/results/traffic/{0:05d}.png".format(i))
