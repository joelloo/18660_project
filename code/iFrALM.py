__all__ = ["iFrALM"]

import numpy as np
from math import *
from fralm import fralm

def ifralm(M, fixed_rank = 1, lam = 0.003, magnitude = 1e-6, batch_size = 5) :
    shape = M.shape
    epoch_num = int(ceil(shape[1] / batch_size))
    if epoch_num < 1 :
        A, E = fralm(M, fixed_rank, lam, magnitude)
        return A, E
    else :
        A_last_batch, E_last_batch = fralm(M[:, :batch_size], fixed_rank, lam, magnitude)
    print("position 1: code went here")
    U_last_batch, s_last_batch, V_last_batch = np.linalg.svd(A_last_batch, full_matrices = True)
    n = batch_size
    batch_inc_order = 1

    for i in range(1, epoch + 1) :
        if n + batch_size > shape[1] :
            l = shape[1] - n
        else :
            l = batch_size
        D = M[:, n : n + l]
        sgn = np.sign(D)
        scl = np.maximum(np.linalg.norm(D, 2), np.linalg.norm(D, np.inf) / lam)
        
        A = np.zeros((shape[0], l))
        E = np.zeros((shape[0], l))
        Y = sgn / scl
        A_last_iter = A
        E_last_iter = E
    
        # parameters from the paper
        s = np.linalg.svd(Y, full_matrices = False, compute_uv = False)
        mu = 0.5 / s[0]
        pho = 6

        in_stop = magnitude * np.linalg.norm(D, 'fro')
        out_stop = in_stop / 10

        in_converge = False
        out_converge = False
        
        #iFrALM
        epoch = 0
        print("position 2: code went here")
        while not out_converge:
            while not in_converge:
                print("position 3: code went here")
                U, s, V = isvd(U_last_batch, s_last_batch, V_last_batch, D)
                print("position 4: code went here")
                A = np.dot(U * s, V[:, n : n + l])
                E = soft_threshhold(D - A + Y / mu, lam / mu)
                in_converge = np.linalg.norm(A - A_last_iter, 'fro') < in_stop and np.linalg.norm(E - E_last_iter, 'fro') < in_stop
                # print("position 3: code went here")
                print("A convergence: " + repr(np.linalg.norm(A - A_last_iter, 'fro')))
                print("E convergence: " + repr(np.linalg.norm(E - E_last_iter, 'fro')))
                A_last_iter = A
                E_last_iter = E

            out_error = np.linalg.norm(D - A - E, 'fro')
            out_converge =  out_error < out_stop
            print("incremental batch order: " + repr(np.linalg.norm(D - A - E, 'fro')) + "epoch : " + repr(epoch) + "error : " + repr(out_error))

            U_tilde, s_tilde, V_tilde = np.linalg.svd(np.transpose(np.transpose(V[:, 1:n] * s)))
            U_last_batch = np.dot(U, U_tilde)
            s_last_batch = s_tilde
            V_last_batch = V_tilde
            Y = Y + mu * (D - A - E)
            mu = pho * mu
            epoch = epoch + 1

            
        U_last_batch = U
        s_last_batch = s
        V_last_batch = V
        n = n + l
    return A, E



def isvd(P_k, s_k, QH_k, D) :
	# input full version of SVD: P_k, s_K, QH_k
	# return full version of SVD: U_i, s_i, V_i
    m, n = D.shape
    k = min(m, n)
    M = np.dot((np.eye(m) - np.dot(P_k, np.transpose(P_k))), D)
    P_k_hat, R = np.linalg.qr(M, mode = 'reduced')
    B = np.zeros((len(s_k) + R.shape[0], len(s_k) + D.shape[1]))
    B[:len(s_k), :len(s_k)] = np.diag(s_k)
    B[:len(s_k), len(s_k): -1]
    B_hat1 = np.concatenate((np.diag(s_k), np.dot(np.transpose(P_k), D)), axis = 1)
    B_hat2 = np.concatenate((np.zeros((k, k)), R), axis = 1)
    B_hat = np.concatenate((B_hat1, B_hat2), axis = 0)
    
    U_k, s_hat, VH_k = np.linalg.svd(B_hat, full_matrices = False)

    U = np.dot(p.concatenate((P_k, P_k_hat)), U_k)
    s = s_hat[:k]
    V1 = np.concatenate((QH_k, np.zeros((n, len*(s_hat) - k))), axis = 1)
    V2 = np.concatenate((np.zeros((len*(s_hat) - k, n)), np.eye(len*(s_hat) - k)), axis = 1)
    V = np.dot(VH_k, np.concatenate((V1, V2), axis = 0))

    return U, s, V


def soft_threshhold(S, tau):
    return np.sign(S) * np.maximum(np.abs(S) - tau, 0)


