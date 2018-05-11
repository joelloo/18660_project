import numpy as np
import cvxpy as cvx
import scipy.io as sio
import matplotlib.pyplot as plt
import time


def orpca(M, lm1, lm2, L0, solve_type, output_err=False):
    m,n = M.shape
    A = np.zeros((n,n))
    B = np.zeros((m,n))
    R = np.zeros((n,n))
    E = np.zeros((m,n))
    L = L0
    err = []
    timing = []

    for i in range(n):
        st = time.process_time()
        #print "Iteration " + str(i)
        z = M[:,i]
        z = np.atleast_2d(z).T
        if solve_type == "cvx":
            r,e = solve_cvx(z,L,lm1,lm2)
        elif solve_type == "altmin":
            r,e,d = solve_altmin(z,L,lm1,lm2)
            err.append(d)
        else:
            #print "Unknown solve type"
            exit(0)
        A = A + np.matmul(r, r.T)
        B = B + np.matmul(z-e, r.T)
        L = basis_update(L,A,B,lm1,lm2)

        R[:,i] = r.flatten()
        E[:,i] = e.flatten()

        end = time.process_time()
        timing.append(end - st)
        

    if output_err:
        lr = np.matmul(L,R)
        return lr, E, np.linalg.norm(M-lr-E, axis=0), err, timing

    return np.matmul(L,R), E

# Update the basis L
def basis_update(L,A,B,lm1,lm2):
    m,n = L.shape
    A = A + lm1*np.identity(n)
    for j in range(n):
        bj = B[:,j]
        aj = A[:,j]
        lj = L[:,j]
        L[:,j] = (bj - np.matmul(L,aj)) / A[j,j] + lj
    return L

# Use CVXPY to solve the subproblem
def solve_cvx(z,L,lm1,lm2):
    m,n = L.shape
    x = cvx.Variable(n)
    e = cvx.Variable(m)
    objective = cvx.Minimize(0.5*cvx.square(cvx.norm(z-L*x-e,2)) + 0.5*lm1*cvx.square(cvx.norm(x,2)) + lm2*cvx.norm(x,1))
    prob = cvx.Problem(objective)
    prob.solve(verbose=True)
    x = np.asarray(x.value)
    e = np.asarray(e.value)
    return x, e

# Use alternating minimization (gradient/prox gradient descent)
def solve_altmin(z,L,lm1,lm2):
    m,n = L.shape
    r = np.zeros((n,1))
    e = np.zeros((m,1))
    prod = np.linalg.inv(np.matmul(L.T,L) + lm1*np.identity(n))
    prod = np.matmul(prod, L.T)
    diff = np.inf
    tol = 1e-5
    diffs = []

    while diff > tol:
        # Minimize over r alone -- since fully differentiable wrt r, we
        # can directly compute the minimum using the derivative
        rprev = r
        r = np.matmul(prod, z-e)
        diff_r = np.linalg.norm(r - rprev)

        # Minimize over e now -- the minimizer is simply the soft
        # thresholding function
        eprev = e
        e = softThresh(z-np.matmul(L,r), lm2)
        diff_e = np.linalg.norm(e - eprev)

        diff = max(diff_e, diff_r)
        diffs.append(diff)

    return r, e, diffs
            

def softThresh(x, lm):
    return np.sign(x) * np.maximum(np.abs(x) - lm, 0)

"""
# Test
data = sio.loadmat('../data/demo_vid.mat')
M = data['M']
ht = np.asscalar(data['vh'])
wd = np.asscalar(data['vw'])

lm1 = 1 / np.sqrt(max(M.shape))
lm2 = lm1
m,n = M.shape
L, S = orpca(M, lm1, lm2, np.random.rand(m,n), "altmin")

for i in range(0,n,5):
    im_lr = L[:,i].reshape(wd,ht).T
    im_sp = S[:,i].reshape(wd,ht).T
    plt.imshow(im_lr, cmap='gray')
    #plt.imshow(im_sp, cmap='gray')
    plt.show()
"""
