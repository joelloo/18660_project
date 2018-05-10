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

        print("PCP status: "  + str(prob.status))
        self.L_ = L.value
        self.S_ = S.value


    # Accelerated proximal gradient descent
    def rpca_apg(self, lm=None, mu0=None, eta=0.9, delt=1e-7, tol=1e-5, conv=1e-7, maxIters=1000):
        D = self.M_

        m,n = D.shape
        mu_k = mu0 if mu0 else 0.99 * np.linalg.norm(D, 2)
        mu_bar = delt * mu_k
        lm = lm if lm else 1. / np.sqrt(m)
        norm_fro = np.linalg.norm(D, 'fro')

        Ak = np.zeros((m,n))
        Ak_1 = Ak
        Ek = np.zeros((m,n))
        Ek_1 = Ek
        tk = 1
        tk_1 = 1
        converged = False
        iters = 0
        Sprev = np.inf

        while iters < maxIters and not converged:
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
            Snext = (np.linalg.norm(S_Anext, 'fro')**2 
                            + np.linalg.norm(S_Enext, 'fro')**2)
            conv_err = abs(Snext-Sprev)
            converged = Snext < tol or conv_err < conv
            Sprev = Snext

            Ak = Anext
            Ak_1 = Ak
            Ek = Enext
            Ek_1 = Ek

            if iters % 100 == 0:
                print("APG: Passed " + str(iters) + " iterations, error: " + str(np.linalg.norm(D - Ak - Ek, 'fro')))
            iters += 1

        print("Final error: " + str(np.linalg.norm(D-Ak-Ek,'fro')) + " | " + str(np.linalg.matrix_rank(Ak)))
        self.L_ = Ak
        self.S_ = Ek


    # Exact ALM, using algorithm, constants described in [Lin,Chen,Ma]

    def rpca_ealm(self, lm=None, mu=None, rho=6, delta=1e-5, deltaProj=1e-5, conv=1e-7, maxIters=100):
        D = self.M_
        m,n = D.shape
        lm = lm if lm else 1. / np.sqrt(m)
        Y = np.sign(D)
        norm_2 = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y, np.inf) / lm
        Y = Y / max(norm_2, norm_inf)
        mu = mu if mu else 0.5 / norm_2
        dnorm = np.linalg.norm(D, 'fro')
        stop  = dnorm * conv

        A = np.zeros((m,n))
        E = np.zeros((m,n))

        iters = 0
        error = np.inf
        stopInner = deltaProj * dnorm

        # while iters < maxIters and np.linalg.norm(D-A-E, 'fro') > stop:
        prev = np.inf
        while True:
            # print iters
            converged = False
            inner_iters = 0

            while not converged:
                mu_k_inv = 1. / mu
                Anext = svt(D - E + mu_k_inv * Y, mu_k_inv)
                Enext = shrinkage(D - Anext + mu_k_inv * Y, lm * mu_k_inv)

                converged = (np.linalg.norm(Anext-A, 'fro') < stopInner
                            and np.linalg.norm(Enext-E, 'fro') < stopInner)
                A = Anext
                E = Enext
                inner_iters += 1

            Y = Y + mu * (D-A-E)
            mu = rho * mu

            error = np.linalg.norm(D-A-E, 'fro')
            curr = error
            conv_err = abs(curr - prev)
            prev = curr
            
            if iters % 1 == 0:
                print("EALM: Passed " + str(iters) + " iterations, error: " + str(error))
            iters += 1

            if iters > maxIters or error < stop or conv_err < conv:
                break

        print("Final error: " + str(np.linalg.norm(D-A-E, 'fro')) + " | " + str(np.linalg.matrix_rank(A)))
        self.L_ = A
        self.S_ = E


    # Inexact ALM, using formulation and constants from [Candes,Li,Ma,Wright,2009]

    def rpca_ialm(self, delta=1e-5, conv=1e-7, mu=None, lm=None):
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

        norm_2 = np.linalg.norm(M, 2)
        norm_inf = np.linalg.norm(M, np.inf) / lm
        norm_fro = np.linalg.norm(M, 'fro')
        # Y = M / max(norm_2, norm_inf)

        iters = 0

        diff = np.inf
        prev = np.inf
        conv_err = np.inf
        while diff > stop and conv_err > conv:
            # Update L
            L = svt(M - S + mu_inv*Y, mu_inv)

            # Update S
            S = shrinkage(M - L + mu_inv*Y, lm_mu_inv)

            # Update Y
            Y = Y + mu * (M - L - S)

            iters += 1
            diff = np.linalg.norm(M-L-S, 'fro')
            nxt = np.linalg.norm(M-L-S, 'fro')
            conv_err = abs(nxt - prev)
            prev = nxt

            if iters % 100 == 0:
                print("Passed " + str(iters) + " iterations: " + str(diff))
                print("IALM: Passed " + str(iters) + " iterations, error: " + str(diff))

        print("Final error: " + str(np.linalg.norm(M-L-S, 'fro')) + " | " + str(np.linalg.matrix_rank(L)))
        self.L_ = L
        self.S_ = S

    def rpca_ialm_v1(self, delta=1e-5, conv=1e-7, mu=None, lm=None, maxIters = 1000):
        M = np.double(self.M_)
        # M.dtype('double')
        m,n = M.shape
        A = np.zeros((m,n))
        E = np.zeros((m,n))
        Y = M
        lm = lm if lm else 1. / np.sqrt(m)
        # mu = mu if mu else m*n / (4*np.linalg.norm(self.M_, 1))

        norm_two = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm( Y.reshape(-1, 1), np.inf) / lm
        dual_norm = np.maximum(norm_two, norm_inf)
        Y = Y / dual_norm

        mu = 1.25 / norm_two
        mu_bar = mu * 1e7;
        mu_inv = 1. / mu
        lm_mu_inv = lm * mu_inv
        rho = 1.5
        
        norm_fro = np.linalg.norm(M, 'fro');
        
        

        iters = 0
        conv_err = np.inf
        # print(norm_fro)
        # print(mu)
        # print(lm)
        # print(norm_two)
        # print(norm_fro)
        while conv_err > conv and iters < maxIters:
            iters = iters + 1
            T = M - A + Y * mu_inv
            E = shrinkage(T, lm_mu_inv)
            # E = np.maximum(T - lm_mu_inv, 0)
            # E = E + np.minimum(T + lm_mu_inv, 0)
            

            U, S, V = np.linalg.svd(M - E + Y * mu_inv, full_matrices = False)
            svp = S > mu_inv

            # print(np.sum(svp))

            A = np.dot(np.dot(U[:, svp], np.diag(S[svp] - mu_inv)), V[svp, :])
            Z = M - A - E
            Y = Y + mu * Z
            # print("iter: " + str(iters) + ", E: " + str(np.linalg.norm(E, 'fro'))
            #     + ", A: " + str(np.linalg.norm(A, 'fro')) + ", mu: " + str(mu)
            #     + ", T: " + str(np.linalg.norm(T, 'fro')) + ", Z: " + str(np.linalg.norm(Z, 'fro')) 
            #     + ", Y: " + str(np.linalg.norm(Y, 'fro'))  )
            mu = min(mu*rho, mu_bar)
            mu_inv = 1. / mu
            lm_mu_inv = lm * mu_inv

            conv_err = np.linalg.norm(Z,'fro') / norm_fro
            if iters % 10 == 0:
                print("Passed " + str(iters) + " iterations: " + str(conv_err))

        self.L_ = A
        self.S_ = E


# Shrinkage operator

def shrinkage(X, tau):
    sgn = np.sign(X)
    rectified = np.maximum(np.abs(X) - tau, 0)
    return np.multiply(sgn, rectified)


# Singular value thresholding operator

def svt(X, tau):
    m,n = X.shape
    minsq = min(m,n)
    U, S, V = np.linalg.svd(X, full_matrices = False)
    thresh = np.maximum(S - tau, 0)
    return np.dot(U * thresh, V)
    # return np.dot(np.dot(U, np.diag(thresh)), V)


