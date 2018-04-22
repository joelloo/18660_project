import numpy as np
import skimage as sim
from math import *

class iFrALM():
	
	def __init__(self, M):
		self.M_ = M

	def iFrALM(self, batchsize = 5, fixed_rank = 1, lam = 0.1):
		#problem to deal with, A = USV rather than USV', you need to deal with some subscript
		m, n = M.shape
		lam = 0.1 #maybe else, to decide
		epoch = ceil(n / batchsize)

		for i in range(:epoch):
			if (i <= epoch - 1) :
				Di = M[:, epoch * batchsize : (epoch + 1) * batchsize]
				Ai = np.zeros(m, batchsize)
				Ei = np.zeros(m, batchsize)
			elif (i == epoch - 1) :
				Di = Di = M[:, epoch * batchsize : n]
				Ai = np.zeros(m, n - epoch * batchsize)
				Ei = np.zeros(m, n - epoch * batchsize)
			sgn_D = np.sign(Di)
			Y = sgn_d / np.maximum(np.norm(sgn_D, 2), -lam * np.norm(sgn_D, inf))
			mu = 0.1 #maybe else, just a guess now
			pho = 0.9 # the same as mu
			while (True) :
				iFrALM_converge = True
				while (True) :
					iSVD_converge = True
					Ui, Si, Vi = iSVD(Ui_1, Si_1, Vi_1, Di)
					Ai = np.matmul(Ui, Si)
					Ai = np.matmul(Ai, np.transpose(Vi[n + 1 : n + l + 1, :]))
					Ei = soft_thresh(lam / mu, Di - A + Y / mu)
					if (iSVD_converge) :
						break
				U_tmp, S_tmp, V_tmp =  np.linalg.svd(np.matmul(Si, np.transpose(Vi[: fixed_rank, :])))
				Ui_1 = np.matmul(Ui, U_tmp)
				Si_1 = S_tmp
				Vi_1 = V_tmp
				Y = Y + mu * (Di - Ai - Ei)
				mu = pho * mu
				if (iFrALM_converge) :
					break;
			Ui_1 = Ui
			Si_1 = Si
			Vi_1 = Vi

		A = np.matmul(Ui, np.matmul(Si, np.transpose(Vi)))




	def iSVD(U, S, V, D) :



	def soft_thresh(eps, M) :
		sgn = np.sign(M)
	    S = np.abs(M) - eps
	    S[S < 0.0] = 0.0
	    return sgn * S







