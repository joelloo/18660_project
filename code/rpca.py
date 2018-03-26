import cvxpy as cvx
import numpy as np
import skimage as sim
import matplotlib.pyplot as plt

# RPCA by Principal Component Pursuit. Based on the convex relaxation
# formulation provided in Robust PCA [Candes, Li, Ma, Wright, 2009].
# Implements various algorithms to optimize the PCP formulation of RPCA

class RPCA():

    def __init__(self, M):
        self.M_ = M


    def rpca_cvx(self):
        m,n = self.M_.shape
        k = 1. / np.sqrt(max(m,n))
        L = cvx.Variable(m,n)
        S = cvx.Variable(m,n)
        obj = cvx.norm(L, "nuc") + k * cvx.norm(S, 1)
        cons = [self.M_ == L + S]
        prob = cvx.Problem(cvx.Minimize(obj), cons)
        prob.solve()

        print "PCP status: "  + str(prob.status)
        self.L_ = L.value
        self.S_ = S.value


    # Accelerated proximal gradient descent

    def rpca_apg(self, lm=None, mu0=None, eta=0.9, delt=1e-5, tol=1e-5, maxIters=1000):
        D = self.M_
        m,n = D.shape
        mu_k = mu0 if mu0 else 0.99 * np.linalg.norm(D, 2)
        mu_bar = delt * mu_k
        lm = lm if lm else 1. / np.sqrt(m)

        Ak = np.zeros((m,n))
        Ak_1 = Ak
        Ek = np.zeros((m,n))
        Ek_1 = Ek
        tk = 1
        tk_1 = 1
        converged = False
        iters = 0

        while iters < maxIters and not converged:
            print iters
            Y_Ak = Ak + (tk_1-1)/tk * (Ak - Ak_1)
            Y_Ek = Ek + (tk_1-1)/tk * (Ek - Ek_1)

            avg = 0.5 * (Y_Ak + Y_Ek - D)
            G_Ak = Y_Ak - avg
            G_Ek = Y_Ek - avg

            Anext = svt(G_Ak, mu_k/2)
            Enext = shrinkage(G_Ek, lm * mu_k / 2)

            tk_1 = tk
            tk = (1 + np.sqrt(4 * tk**2 + 1)) / 2
            mu_k = max(eta * mu_k, mu_bar)
            
            diff = Anext + Enext - Y_Ak - Y_Ek
            S_Anext = 2 * (Y_Ak - Anext) + diff 
            S_Enext = 2 * (Y_Ek - Enext) + diff
            Snext = np.sqrt(np.linalg.norm(S_Anext, 'fro')**2 
                            + np.linalg.norm(S_Enext, 'fro')**2)
            converged = Snext < tol
            Ak = Anext
            Ak_1 = Ak
            Ek = Enext
            Ek_1 = Ek

            iters += 1

        self.L_ = Ak
        self.S_ = Ek


    # Exact ALM, using algorithm, constants described in [Lin,Chen,Ma]

    def rpca_ealm(self, lm=None, mu=None, rho=6, delta=1e-7, deltaProj=1e-6, maxIters=100):
        D = self.M_
        m,n = D.shape
        lm = lm if lm else 1. / np.sqrt(m)
        Y = np.sign(D)
        norm_2 = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y, np.inf) / lm
        Y = Y / max(norm_2, norm_inf)
        mu = mu if mu else 0.5 / norm_2
        dnorm = np.linalg.norm(D, 'fro')

        A = np.zeros((m,n))
        E = np.zeros((m,n))

        iters = 0
        stop = delta * dnorm
        stopInner = deltaProj * dnorm

        while iters < maxIters and np.linalg.norm(D-A-E, 'fro') > stop:
            print iters
            converged = False

            while not converged:
                mu_k_inv = 1. / mu
                Anext = svt(D - E + mu_k_inv * Y, mu_k_inv)
                Enext = shrinkage(D - Anext + mu_k_inv * Y, lm * mu_k_inv)

                converged = (np.linalg.norm(Anext-A, 'fro') < stopInner
                            and np.linalg.norm(Enext-E, 'fro') < stopInner)
                A = Anext
                E = Enext

            Y = Y + mu * (D-A-E)
            mu = rho * mu
            iters += 1

        self.L_ = A
        self.S_ = E


    # Inexact ALM, using formulation and constants from [Candes,Li,Ma,Wright,2009]

    def rpca_ialm(self, delta=1e-7, mu=None, lm=None):
        # Init constants and vars
        M = self.M_
        m,n = M.shape
        L = np.zeros((m,n))
        S = np.zeros((m,n))
        Y = np.zeros((m,n))
        lm = lm if lm else 1. / np.sqrt(np.max(M.shape))
        mu = mu if mu else m*n / (4*np.linalg.norm(self.M_, 1))
        mu_inv = 1. / mu
        lm_mu_inv = lm * mu_inv
        stop = delta * np.linalg.norm(M, 'fro')

        iters = 0
        diff = np.inf
        while diff > stop:
            # Update L
            L = svt(M - S + mu_inv*Y, mu_inv)

            # Update S
            S = shrinkage(M - L + mu_inv*Y, lm_mu_inv)

            # Update Y
            Y = Y + mu * (M - L - S)

            iters += 1
            diff = np.linalg.norm(M-L-S, 'fro')
            if iters % 100 == 0:
                print "Passed " + str(iters) + " iterations: " + str(diff)

        self.L_ = L
        self.S_ = S


# Shrinkage operator

def shrinkage(X, tau):
    sgn = np.sign(X)
    rectified = np.maximum(np.abs(X) - tau, 0)
    return np.multiply(sgn, rectified)


# Singular value thresholding operator

def svt(X, tau):
    m,n = X.shape
    minsq = min(m,n)

    U, S, V = np.linalg.svd(X)
    thresh = np.maximum(S - tau, 0)
    smat = np.zeros((m,n))
    smat[:minsq, :minsq] = np.diag(thresh)

    return np.dot(U, np.dot(smat, V))


