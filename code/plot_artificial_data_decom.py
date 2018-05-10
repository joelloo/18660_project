import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import rand
# from rpca import *


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
    return np.dot(np.dot(U, np.diag(thresh)), V)

class TestAlg():
    def __init__(self, shape, rank):
        a = np.random.random(size=(shape[0], rank))
        q, _ = np.linalg.qr(a)
        self.shape_ = shape
        self.rank_ = rank
        self.A_ = np.zeros(shape)
        for i in range(shape[1]):
            self.A_[:, i] = np.dot(q, np.random.dirichlet(np.ones(rank), size = 1).reshape(rank, 1))[:, 0]
        self.E_ = rand(shape[0], shape[1], density = 0.1,format= 'coo', dtype=None).todense()
        self.D_ = self.A_ + self.E_
        self.normD = np.linalg.norm(self.D_, 'fro')
        self.normA = np.linalg.norm(self.A_, 'fro')
        self.normE = np.linalg.norm(self.E_, 'fro')

        self.iter_ = 0
        self.err_ = np.zeros((1, 10000))
        self.rank_ = np.zeros((1, 10000))
        self.err_arr_A_ = np.zeros((1, 10000))
        self.err_arr_E_ = np.zeros((1, 10000))
        self.diff_A = np.zeros((1, 10000))
        self.diff_E = np.zeros((1, 10000))
        self.diff_A_prime = np.zeros((1, 10000))
        self.diff_E_prime = np.zeros((1, 10000))

    def plot(self, alg = 'all'):
        if alg == 'all':
            self.rpca_ealm()
            plt.figure(1)
            plt.plot(range(self.iter_  + 1), self.err_[0, : self.iter_ + 1], 'r', label = 'norm(D - A\' - E\') / norm(D)')
            plt.plot(range(self.iter_  + 1), self.rank_[0, : self.iter_ + 1], 'k', label = 'rank of A')
            plt.plot(range(self.iter_  + 1), self.err_arr_A_[0, : self.iter_ + 1], 'g-.', label = 'norm(A - A\') / norm(A)')
            plt.plot(range(self.iter_  + 1), self.err_arr_E_[0, : self.iter_ + 1], 'b--', label = 'norm(E - E\') / norm(E)')
            plt.plot(range(1, self.iter_  + 1), self.diff_A[0, 1 : self.iter_ + 1], 'y', label = 'norm(A\' - A\' in last iteration)')
            plt.plot(range(1, self.iter_  + 1), self.diff_E[0, 1 : self.iter_ + 1], 'c', label = 'norm(E\' - E\' in last iteration)')
            # plt.plot(range(self.iter_  + 1), self.diff_A_prime[0, : self.iter_ + 1], 'k', label = 'D - E\' - A')
            # plt.plot(range(self.iter_  + 1), self.diff_E_prime[0, : self.iter_ + 1], 'm', label = 'D - A\' - E')

            plt.yscale('log')
            plt.title('Artificial matrix recovery (size: 1000 by 100, rank: 5) using ealm')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.legend()

            self.reset()
            self.rpca_ialm()

            plt.figure(2)
            plt.plot(range(self.iter_  + 1), self.err_[0, : self.iter_ + 1], 'r', label = 'norm(D - A\' - E\') / norm(D)')
            plt.plot(range(self.iter_  + 1), self.rank_[0, : self.iter_ + 1], 'k', label = 'rank of A')
            plt.plot(range(self.iter_  + 1), self.err_arr_A_[0, : self.iter_ + 1], 'g-.', label = 'norm(A - A\') / norm(A)')
            plt.plot(range(self.iter_  + 1), self.err_arr_E_[0, : self.iter_ + 1], 'b--', label = 'norm(E - E\') / norm(E)')
            plt.plot(range(1, self.iter_  + 1), self.diff_A[0, 1 : self.iter_ + 1], 'y', label = 'norm(A\' - A\' in last iteration)')
            plt.plot(range(1, self.iter_  + 1), self.diff_E[0, 1 : self.iter_ + 1], 'c', label = 'norm(E\' - E\' in last iteration)')
            # plt.plot(range(self.iter_  + 1), self.diff_A_prime[0, : self.iter_ + 1], 'k', label = 'D - E\' - A')
            # plt.plot(range(self.iter_  + 1), self.diff_E_prime[0, : self.iter_ + 1], 'm', label = 'D - A\' - E')

            plt.yscale('log')
            plt.title('Artificial matrix recovery (size: 1000 by 100, rank: 5) using ialm')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.legend()


            self.reset()
            self.rpca_apg()

            plt.figure(3)
            plt.plot(range(self.iter_  + 1), self.err_[0, : self.iter_ + 1], 'r', label = 'norm(D - A\' - E\') / norm(D)')
            plt.plot(range(self.iter_  + 1), self.rank_[0, : self.iter_ + 1], 'k', label = 'rank of A')
            plt.plot(range(self.iter_  + 1), self.err_arr_A_[0, : self.iter_ + 1], 'g-.', label = 'norm(A - A\') / norm(A)')
            plt.plot(range(self.iter_  + 1), self.err_arr_E_[0, : self.iter_ + 1], 'b--', label = 'norm(E - E\') / norm(E)')
            plt.plot(range(1, self.iter_  + 1), self.diff_A[0, 1 : self.iter_ + 1], 'y', label = 'norm(A\' - A\' in last iteration)')
            plt.plot(range(1, self.iter_  + 1), self.diff_E[0, 1 : self.iter_ + 1], 'c', label = 'norm(E\' - E\' in last iteration)')
            # plt.plot(range(self.iter_  + 1), self.diff_A_prime[0, : self.iter_ + 1], 'k', label = 'D - E\' - A')
            # plt.plot(range(self.iter_  + 1), self.diff_E_prime[0, : self.iter_ + 1], 'm', label = 'D - A\' - E')

            plt.title('Artificial matrix recovery (size: 1000 by 100, rank: 5) using apg')
            plt.yscale('log')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.legend()
            
            plt.show()

    def reset(self):
        self.err_arr_A_ = np.zeros((1, 10000))
        self.err_arr_E_ = np.zeros((1, 10000))
        self.err_ = np.zeros((1, 10000))
        self.diff_A = np.zeros((1, 10000))
        self.diff_E = np.zeros((1, 10000))
        self.iter_ = 0

    def update(self, A, E, A_diff, E_diff, iter):
        self.iter_ = iter
        self.rank_[0, iter] = np.linalg.matrix_rank(A)
        self.err_[0, iter] = np.linalg.norm(self.D_ - A - E, 'fro') / self.normD
        self.err_arr_A_[0, iter] = np.linalg.norm(self.A_ - A, 'fro') / self.normA
        self.err_arr_E_[0, iter] = np.linalg.norm(self.E_ - E, 'fro') / self.normE
        self.diff_A_prime[0, iter] = np.linalg.norm(self.D_ - A - self.E_, 'fro') / self.normA
        self.diff_E_prime[0, iter] = np.linalg.norm(self.D_ - E - self.A_, 'fro') / self.normE 
        if A_diff is not None :
            # self.diff_A[0, iter] = np.linalg.norm(A_diff, 'fro') / self.normA
            # self.diff_E[0, iter] = np.linalg.norm(E_diff, 'fro') / self.normE
            self.diff_A[0, iter] = np.linalg.norm(A_diff, 'fro')
            self.diff_E[0, iter] = np.linalg.norm(E_diff, 'fro')

    def rpca_apg(self, lm=None, mu0=None, eta=0.9, delta=1e-10, tol=1e-5, maxIters=1000):
        D = self.D_
        m,n = D.shape
        mu_k = 0.99 * np.linalg.norm(D, 2)
        mu_bar = delta * mu_k
        lm = 1. / np.sqrt(m)
        norm_fro = np.linalg.norm(D, 'fro')
        Ak = np.zeros((m,n))
        Ak_1 = Ak
        Ek = np.zeros((m,n))
        Ek_1 = Ek
        tk = 1
        tk_1 = 1
        converged = False
        iters = 0
        error = np.inf
        
        self.update(Ak, Ek, None, None, iters)

        while iters < maxIters and not converged:
            iters += 1
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
            Snext = np.sqrt(np.linalg.norm(S_Anext, 'fro')**2 + np.linalg.norm(S_Enext, 'fro')**2)
            # converged = Snext < tol
            error = np.linalg.norm(D - Anext - Enext, 'fro') / norm_fro
            converged = error < delta
            Ak = Anext
            Ek = Enext

            self.update(Ak, Ek, Ak - Ak_1, Ek - Ek_1, iters)

            Ak_1 = Ak
            Ek_1 = Ek

            if iters % 50 == 0:
                print "APG: Passed " + str(iters) + " iterations, error: " + str(error)

    def rpca_ealm(self, lm=None, mu=None, rho=6, delta=1e-10, deltaProj=1e-6, maxIters=100): 
        D = self.D_
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

        A_last = A
        E_last = E

        iters = 0

        self.update(A, E, None, None, iters)

        error = np.inf
        stopInner = deltaProj * dnorm

        while iters < maxIters and error > delta:
            converged = False
            while not converged:
                mu_k_inv = 1. / mu
                Anext = svt(D - E + mu_k_inv * Y, mu_k_inv)
                Enext = shrinkage(D - Anext + mu_k_inv * Y, lm * mu_k_inv)

                converged = (np.linalg.norm(Anext-A, 'fro') < stopInner and np.linalg.norm(Enext-E, 'fro') < stopInner)
                A = Anext
                E = Enext

            Y = Y + mu * (D-A-E)
            mu = rho * mu
            error = np.linalg.norm(D-A-E, 'fro') / dnorm

            iters += 1

            self.update(A, E, A - A_last, E - E_last, iters)

            A_last = A 
            E_last = E

            if iters % 2 == 0:
                print "EALM: Passed " + str(iters) + " iterations, error: " + str(error)

    def rpca_ialm(self, delta=1e-10, mu=None, lm=None, maxIters = 1000):
        D = np.double(self.D_)
        m,n = D.shape
        
        Y = D
        lm = lm if lm else 1. / np.sqrt(m)
        norm_two = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y.reshape(-1, 1), np.inf) / lm
        dual_norm = np.maximum(norm_two, norm_inf)
        Y = Y / dual_norm

        mu = 1.25 / norm_two
        mu_bar = mu * 1e7;
        mu_inv = 1. / mu
        lm_mu_inv = lm * mu_inv
        rho = 1.5
        
        norm_fro = np.linalg.norm(D, 'fro')

        A = np.zeros((m,n))
        E = np.zeros((m,n))

        A_last = A
        E_last = E
        
        iters = 0
        self.update(A, E, None, None, iters)

        error = np.inf
        while error > delta and iters < maxIters:
            E = shrinkage(D - A + Y * mu_inv, lm_mu_inv)
            U, S, V = np.linalg.svd(D - E + Y * mu_inv, full_matrices = False)
            svp = S > mu_inv

            A = np.dot(np.dot(U[:, svp], np.diag(S[svp] - mu_inv)), V[svp, :])
            Z = D - A - E
            Y = Y + mu * Z

            mu = min(mu*rho, mu_bar)
            mu_inv = 1. / mu
            lm_mu_inv = lm * mu_inv

            error = np.linalg.norm(Z,'fro') / norm_fro

            iters += 1

            self.update(A, E, A - A_last, E - E_last, iters)
            
            A_last = A
            E_last = E

            if iters % 10 == 0:
                print("IALM: Passed " + str(iters) + " iterations: " + str(error))
            

shape = [1000, 100]
rank = 5
sample = TestAlg(shape, rank)
sample.plot()


