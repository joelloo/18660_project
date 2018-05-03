import numpy as np
from math import *

# This method is extracted from Wee Kheng Leow's paper: Background Recovery by Fixed-fixed_rank Robust
# Principal Component Analysis

def fralm(D, fixed_rank = 1, lam = 0.003, inner_error = 1e-3, outer_error = 1e-7):
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

	in_stop = inner_error * np.linalg.norm(D, 'fro')
	out_stop = outer_error * np.linalg.norm(D, 'fro')

	in_converge = False
	out_converge = False

	iters = 0;

	while not out_converge:
		while not in_converge:
			U, s, V = np.linalg.svd(D - E + Y / mu, full_matrices = False)
			s = soft_threshhold(s, 1 / mu)
			if np.sum(s > 0) <= fixed_rank :
				A = np.dot(U * s, V)
			else :
				s[fixed_rank:] = 0
				A = np.dot(U * s, V)
			E = soft_threshhold(D - A + Y / mu, lam / mu)
			print(np.linalg.norm(A - A_last, 'fro'))
			print(np.linalg.norm(E - E_last, 'fro'))
			in_converge = np.linalg.norm(A - A_last, 'fro') < in_stop and \
							np.linalg.norm(E - E_last, 'fro') < in_stop
			A_last = A
			E_last = E
			
		err = np.linalg.norm(D - A - E, 'fro')
		in_converge = False
		out_converge = err < out_stop
		print('Iterations: ' + repr(iters) + 'Error:' + repr(err))
		Y = Y + mu * (D - A - E)
		mu = pho * mu
		iters = iters + 1

	return A, E



def soft_threshhold(S, tau):
	return np.sign(S) * np.maximum(np.abs(S) - tau, 0)
