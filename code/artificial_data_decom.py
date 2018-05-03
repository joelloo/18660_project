import random
import numpy as np
import scipy


shape = [10000, 100]
rank = 5
A = np.zeros(shape)

a = np.random.random(size=(shape[0], rank))
q = scipy.linalg.orth(a)

for i in range(shape[1]):
	A[:, i] = np.dot(q * np.random.dirichlet(np.ones(rank), size = 1).reshape(rank, 1))

E = 