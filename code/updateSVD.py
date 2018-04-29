import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.utils.extmath import randomized_svd


# Incremental SVD. Returns u,s,vh such that u@s@vh = [D d]
# Takes in thin SVD, i.e. u@s@vh = D such that u->mxn, s->nxn, vh->nxn
def incrSVD(d, U0, S0, V0, full=False):
    d = np.atleast_2d(d).T
    r = len(S0)
    S0 = np.diag(S0)

    x = np.matmul(U0.T, d)
    z = d - np.matmul(U0, x)
    rho = np.linalg.norm(z)
    p = z / rho
    test = np.zeros((1,r))
    middle = np.vstack([np.hstack([S0, x]), 
                    np.hstack([np.zeros((1,r)), np.atleast_2d(rho)])])
    G, Shat, HT = np.linalg.svd(middle)
    H = HT.T

    if full:
        vh,vw = V0.shape
        U1 = np.matmul(np.hstack([U0, p]), G)
        S1 = Shat
        V1 = np.vstack([np.hstack([V0, np.zeros((vh,1))]),
                    np.hstack([np.zeros((1,vw)), np.atleast_2d(1)])])
        V1 = np.matmul(V1, H)
        return U1, S1, V1.T
    else:
        U1 = np.matmul(U0, G[:r,:r]) + np.matmul(p, np.atleast_2d(G[r,:r]))
        S1 = Shat[:r]
        V1 = np.vstack([np.matmul(V0, H[:r,:r]), H[r,:r]])
        return U1, S1, V1.T
    

# Decremental SVD
def decrSVD(idx, D, U0, S0, V0):
    m,l1 = D.shape
    r = len(S0)
    S0 = np.diag(S0)
    e = np.zeros((1,l1)).T
    e[idx,0] = 1.
    d = np.atleast_2d(D[:,idx]).T

    y = np.atleast_2d(V0[-1,:]).T
    z_y = e - np.matmul(V0, y)
    rho_y = np.linalg.norm(z_y)
    q = z_y / rho_y
    x = -np.matmul(S0, y)

    middle = np.hstack([S0+np.matmul(x,y.T), rho_y*x])
    middle = np.vstack([middle, np.zeros((1,r+1))])
    G, Shat, HT = np.linalg.svd(middle)
    H = HT.T

    U1 = np.matmul(U0, G[:r,:r])
    S1 = Shat[:r]
    V1 = np.matmul(V0, H[:r,:r]) + np.matmul(q, np.atleast_2d(H[r,:r]))
    return U1, S1, V1.T
    

# Replaces last column in SVD
def replSVD(dhat, U0, S0, V0):
    l1,_ = V0.shape
    r = len(S0)
    S0 = np.diag(S0)
    e = np.zeros((1,l1)).T
    e[-1,0] = 1.
    dhat = np.atleast_2d(dhat).T

    y = np.atleast_2d(V0[-1,:]).T
    z_y = e - np.matmul(V0, y)
    rho_y = np.linalg.norm(z_y)
    q = z_y / rho_y

    x = np.matmul(U0.T, dhat) - np.matmul(S0, y)
    z_x = dhat - np.matmul(U0, np.matmul(U0.T, dhat))
    rho_x = np.linalg.norm(z_x)
    p = z_x / rho_x

    mid1 = np.hstack([S0 + np.matmul(x,y.T), rho_y*x])
    mid2 = np.hstack([rho_x*y.T, np.atleast_2d(rho_x*rho_y)])
    middle = np.vstack([mid1, mid2]) 
    G, Shat, HT = np.linalg.svd(middle)
    H = HT.T

    U1 = np.matmul(U0, G[:r,:r]) + np.matmul(p, np.atleast_2d(G[r,:r]))
    S1 = Shat[:r]
    V1 = np.matmul(V0, H[:r,:r]) + np.matmul(q, np.atleast_2d(H[r,:r]))
    return U1, S1, V1.T
    

# Test SVD editing functions
"""
mat = np.random.rand(7,4)
ug, sg, vg = np.linalg.svd(mat, full_matrices=False)
ui, si, vi = np.linalg.svd(mat[:,:-1], full_matrices=False)
ut, st, vt = incrSVD(mat[:,-1], ui, si, vi.T, full=True)

rg = np.matmul(ug, np.matmul(np.diag(sg), vg))
rt = np.matmul(ut, np.matmul(np.diag(st), vt))
if np.allclose(rg, rt):
    print "Incremental - passed"
else:
    print "Incremental - differs"

mat = np.random.rand(10, 4)
mat = np.hstack([mat,2*mat])
ug, sg, vg = randomized_svd(mat[:,:-1], n_components=4) 
ui, si, vi = randomized_svd(mat, n_components=4)
ut, st, vt = decrSVD(-1, mat, ui, si, vi.T)

rg = np.matmul(ug, np.matmul(np.diag(sg), vg))
rt = np.matmul(ut, np.matmul(np.diag(st), vt))

if np.allclose(rg, rt[:,:-1]):
    print "Decremental - passed"
else:
    print "Decremental - differs"

mat = np.random.rand(7,2)
mat = np.hstack([mat, 2*mat])
dhat = np.random.rand(7)
matrep = np.hstack([mat[:,:-1], np.atleast_2d(dhat).T])
ug, sg, vg = randomized_svd(matrep, n_components=2)
ui, si, vi = randomized_svd(mat, n_components=2)
ut, st, vt = replSVD(dhat, ui, si, vi.T)

rg = np.matmul(ug, np.matmul(np.diag(sg), vg))
rt = np.matmul(ut, np.matmul(np.diag(st), vt))
if np.allclose(rg, rt):
    print "Replace - passed"
else:
    print "Replace - differs"
"""
