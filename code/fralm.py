import numpy as np
from math import *

# This method is extracted from Wee Kheng Leow's paper: Background Recovery by Fixed-fixed_rank Robust
# Principal Component Analysis

def fralm(D, fixed_rank = 1, lam = 0.003, magnitude = 1e-6):
	shp = D.shape
	sgn = np.sign(D)
	scl = np.maximum(np.linalg.norm(D, 2), np.linalg.norm(D, np.inf) / lam)
	
	A = np.zeros(shp)
	E = np.zeros(shp)
	Y = sgn / scl
	A_last = A
	E_last = E
	
	# parameters from the paper
	s = np.linalg.svd(Y, full_matrices = False, compute_uv = False)
	mu = 0.5 / s[0]
	print(mu)
	pho = 6

	in_stop = magnitude * np.linalg.norm(D, 'fro')
	out_stop = in_stop / 10

	in_converge = False
	out_converge = False

	while not out_converge:
		while not in_converge:
			U, s, V = np.linalg.svd(D - E + Y / mu, full_matrices = False)
			s = soft_threshhold(s, 1 / mu)
			if np.sum(s > 0) < fixed_rank :
				A = np.dot(U * s, V)
			else :
				s[fixed_rank: -1] = 0
				A = np.dot(U * s, V)
			E = soft_threshhold(D - A + Y / mu, lam / mu)
			in_converge = np.linalg.norm(A - A_last, 'fro') < in_stop and np.linalg.norm(E - E_last, 'fro') < in_stop
			A_last = A
			E_last = E
		in_converge = False
		out_converge = np.linalg.norm(D - A - E, 'fro') < out_stop
		print('require' + repr(out_stop))
		print('D convergence' + repr(np.linalg.norm(D - A - E, 'fro')) + repr(out_converge))
		Y = Y + mu * (D - A - E)
		mu = pho * mu

	return A, E



def soft_threshhold(S, tau):
	return np.sign(S) * np.maximum(np.abs(S) - tau, 0)


