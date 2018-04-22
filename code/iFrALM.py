__all__ = ["iFrALM"]

import numpy as np
from math import *


def iFrALM(M, batchsize = 5, fixed_rank = 1, lam = 0.1, inner_converge = 1e-6, outer_converge = 1e-6):
	m_, n_ = M.shape
	lam = 0.1 					# may be something else, to decide
	epoch = ceil(n_ / batchsize)
	Ui_1 = None
	si_1 = None
	Vi_1 = None
	Ai_1 = None
	n = 0;
	l = 0;
	for i in range(int(epoch)):
		# find batchsize for this epoch
		if (i < epoch - 1) :
			l = batchsize
		elif (i == epoch - 1) :
			l = n_ - n

		mu = 0.1 				# the same as lam
		pho = 0.9 				# the same as lam

        print l
        Di = M[:, n : n + l]
        Ai = np.zeros((m_, l))
        Ei = np.zeros((m_, l))

        Mi = np.concatenate((Ai_1, Di), axis = 1)
        norm_Mi = np.sum(Mi ** 2)
        norm_Di = np.sum(Mi ** 2)
        
        sgn_D = np.sign(Di)
        sgn_M = np.sign(Mi)
        Y_Di = sgn_d / np.maximum(np.norm(sgn_D, 2), - lam * np.norm(sgn_D, inf))
        Y_Mi = sgn_M / np.maximum(np.norm(sgn_M, 2), - lam * np.norm(sgn_M, inf))

		# iFrALM loop
        while (True) :
            iFrALM_converge = True
			# iSVD loop
            while (True) :
                Ui, si, Vi = iSVD(Ui_1, Si_1, Vi_1, Di - Ei + Y_Di / mu)
                Ai = np.dot(Ui, np.transpose(np.transpose(Vi[:fixed_rank, n : n + l] * si)))
                Ei = soft_thresh(lam / mu, Di - Ai + Y_Di / mu)
                error_inner = np.sqrt(np.sum((Di - Ai - Ei) ** 2) / norm_Di)
                if (error < inner_converge) :
                    break
            U_tmp, s_tmp, V_tmp =  np.linalg.svd(np.transpose(np.transpose(Vi[: fixed_rank , : n]) * si))
            Ui_1 = np.dot(Ui, U_tmp)
            si_1 = s_tmp
            Vi_1 = V_tmp
            A = np.dot(Ui , np.transpose(np.transpose(Vi[:fixed_rank, :] * si)))
            E = soft_thresh(lam / mu, Mi - A + Y_Mi / mu)
            Y_Mi = Y_Mi + mu * (Mi - A - E)
            Y_Di = Y_Di + mu * (Di - Ai - Ei)
            mu = pho * mu
            error = np.sqrt(np.sum((Mi - A - E) ** 2) / norm_Mi) 
            if (error < outer_converge) :
                break;

		# batch update
        n = n + l
        Ui_1 = Ui
        si_1 = si
        Vi_1 = Vi
    # return A, E



def iSVD(P_k, s_k, QH_k, D) :
	# input full version of SVD: P_k, s_K, QH_k
	# return full version of SVD: U_i, s_i, V_i
    m, n = D.shape
    k = min(m, n)
    M = np.dot((np.eye(m) - np.dot(P_k, np.transpose(P_k))), D)
    P_k_hat, R = np.linalg.qr(M, mode = 'reduced')

    B_hat1 = np.concatenate((np.diag(s_k), np.dot(np.transpose(P_k), D)), axis = 1)
    B_hat2 = np.concatenate((np.zeros((k, k)), R), axis = 1)
    B_hat = np.concatenate((B_hat1, B_hat2), axis = 0)
    
    U_k, s_hat, VH_k = np.linalg.svd(B_hat, full_matrices = False)

    U_i = np.dot(p.concatenate((P_k, P_k_hat)), U_k)
    s_i = s_hat[:k]
    V_i_1 = np.concatenate((QH_k, np.zeros((n, len*(s_hat) - k))), axis = 1)
    V_i_2 = np.concatenate((np.zeros((len*(s_hat) - k, n)), np.eye(len*(s_hat) - k)), axis = 1)
    Vi = np.dot(VH_k, np.concatenate((V_i_1, V_i_2), axis = 0))

    return U_i, s_i, V_i


def soft_thresh(eps, M) :
    print "eps: ", eps
    sgn = np.sign(M)
    S = np.abs(M)- eps
    S[S < 0.0] = 0.0
    return sgn * S







