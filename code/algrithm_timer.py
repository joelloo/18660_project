from skimage.transform import *
from skimage import color
from skimage import io
from timeit import timeit
from rpca import *

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import time



def downscale(M, h, w, scale = 0.5):
	size = M.shape
	M0 = rescale(M[:, 0].reshape(w, h),scale)
	h_out = M0.shape[1]
	w_out = M0.shape[0]
	M_out = np.zeros((h_out * w_out, size[1]))
	for i in range(size[1]):
		Mi = rescale(M[:, i].reshape(w, h),scale)
		M_out[:, i] = Mi.reshape(-1, 1)[:, 0]
	return M_out, h_out, w_out

def wrapper(func, *args) :
	def wrapped() :
		return func(*args)
	return wrapped

test_number = 1

demo_data = sio.loadmat('../data/demo_vid.mat')
demo_M_1 = demo_data['M']
demo_h_1 = np.asscalar(demo_data['vh'])
demo_w_1 = np.asscalar(demo_data['vw'])

demo_M_2, demo_h_2, demo_w_2 = downscale(demo_M_1, demo_h_1, demo_w_1, scale = 0.8)
demo_M_3, demo_h_3, demo_w_3 = downscale(demo_M_1, demo_h_1, demo_w_1, scale = 0.6)
demo_M_4, demo_h_4, demo_w_4 = downscale(demo_M_1, demo_h_1, demo_w_1, scale = 0.4)
demo_M_5, demo_h_5, demo_w_5 = downscale(demo_M_1, demo_h_1, demo_w_1, scale = 0.2)

demo_time_cost = np.zeros((5,3))
demo_rpca_res1 = RPCA(demo_M_1)
demo_rpca_res2 = RPCA(demo_M_2)
demo_rpca_res3 = RPCA(demo_M_3)
demo_rpca_res4 = RPCA(demo_M_4)
demo_rpca_res5 = RPCA(demo_M_5)


# demo_time_cost[4, 0] = timeit(demo_rpca_res1.rpca_apg, number = test_number) / test_number
# demo_time_cost[3, 0] = timeit(demo_rpca_res2.rpca_apg, number = test_number) / test_number
# demo_time_cost[2, 0] = timeit(demo_rpca_res3.rpca_apg, number = test_number) / test_number
# demo_time_cost[1, 0] = timeit(demo_rpca_res4.rpca_apg, number = test_number) / test_number
# demo_time_cost[0, 0] = timeit(demo_rpca_res5.rpca_apg, number = test_number) / test_number

# demo_time_cost[4, 1] = timeit(demo_rpca_res1.rpca_ealm, number = test_number) / test_number
# demo_time_cost[3, 1] = timeit(demo_rpca_res2.rpca_ealm, number = test_number) / test_number
# demo_time_cost[2, 1] = timeit(demo_rpca_res3.rpca_ealm, number = test_number) / test_number
# demo_time_cost[1, 1] = timeit(demo_rpca_res4.rpca_ealm, number = test_number) / test_number
# demo_time_cost[0, 1] = timeit(demo_rpca_res5.rpca_ealm, number = test_number) / test_number

# demo_time_cost[4, 2] = timeit(demo_rpca_res1.rpca_ialm, number = test_number) / test_number
# demo_time_cost[3, 2] = timeit(demo_rpca_res2.rpca_ialm, number = test_number) / test_number
# demo_time_cost[2, 2] = timeit(demo_rpca_res3.rpca_ialm, number = test_number) / test_number
# demo_time_cost[1, 2] = timeit(demo_rpca_res4.rpca_ialm, number = test_number) / test_number
# demo_time_cost[0, 2] = timeit(demo_rpca_res5.rpca_ialm, number = test_number) / test_number

esca_data = sio.loadmat('../data/escalator_data.mat')
esca_M_1 = esca_data['X']
esca_h_1 = np.asscalar(esca_data['m'])
esca_w_1 = np.asscalar(esca_data['n'])

esca_M_2, esca_h_2, esca_w_2 = downscale(esca_M_1, esca_h_1, esca_w_1, scale = 0.8)
esca_M_3, esca_h_3, esca_w_3 = downscale(esca_M_1, esca_h_1, esca_w_1, scale = 0.6)
esca_M_4, esca_h_4, esca_w_4 = downscale(esca_M_1, esca_h_1, esca_w_1, scale = 0.4)
esca_M_5, esca_h_5, esca_w_5 = downscale(esca_M_1, esca_h_1, esca_w_1, scale = 0.2)

esca_time_cost = np.zeros((5,3))
esca_rpca_res1 = RPCA(esca_M_1)
esca_rpca_res2 = RPCA(esca_M_2)
esca_rpca_res3 = RPCA(esca_M_3)
esca_rpca_res4 = RPCA(esca_M_4)
esca_rpca_res5 = RPCA(esca_M_5)

# esca_time_cost[4, 0] = timeit(esca_rpca_res1.rpca_apg, number = test_number) / test_number
esca_time_cost[3, 0] = timeit(esca_rpca_res2.rpca_apg, number = test_number) / test_number
esca_time_cost[2, 0] = timeit(esca_rpca_res3.rpca_apg, number = test_number) / test_number
esca_time_cost[1, 0] = timeit(esca_rpca_res4.rpca_apg, number = test_number) / test_number
esca_time_cost[0, 0] = timeit(esca_rpca_res5.rpca_apg, number = test_number) / test_number


# esca_time_cost[4, 1] = timeit(esca_rpca_res1.rpca_ealm, number = test_number) / test_number
esca_time_cost[3, 1] = timeit(esca_rpca_res2.rpca_ealm, number = test_number) / test_number
esca_time_cost[2, 1] = timeit(esca_rpca_res3.rpca_ealm, number = test_number) / test_number
esca_time_cost[1, 1] = timeit(esca_rpca_res4.rpca_ealm, number = test_number) / test_number
esca_time_cost[0, 1] = timeit(esca_rpca_res5.rpca_ealm, number = test_number) / test_number

# esca_time_cost[4, 2] = timeit(esca_rpca_res1.rpca_ialm, number = test_number) / test_number
esca_time_cost[3, 2] = timeit(esca_rpca_res2.rpca_ialm, number = test_number) / test_number
esca_time_cost[2, 2] = timeit(esca_rpca_res3.rpca_ialm, number = test_number) / test_number
esca_time_cost[1, 2] = timeit(esca_rpca_res4.rpca_ialm, number = test_number) / test_number
esca_time_cost[0, 2] = timeit(esca_rpca_res5.rpca_ialm, number = test_number) / test_number
# esca_time_cost[0, 2] = timeit(esca_rpca_res5.rpca_ialm, number = test_number) / test_number

face_res1 = io.imread("../data/yalefaces/subject01.leftlight")
face_res1 = color.rgb2gray(face_res1);
face_res2 = rescale(face_res1, 0.8)
face_res3 = rescale(face_res1, 0.6)
face_res4 = rescale(face_res1, 0.4)
face_res5 = rescale(face_res1, 0.2)

face_time_cost = np.zeros((5,3))
face_rpca_res1 = RPCA(face_res1)
face_rpca_res2 = RPCA(face_res2)
face_rpca_res3 = RPCA(face_res3)
face_rpca_res4 = RPCA(face_res4)
face_rpca_res5 = RPCA(face_res5)

# face_time_cost[4, 0] = timeit(face_rpca_res1.rpca_apg, number = test_number) / test_number
# face_time_cost[3, 0] = timeit(face_rpca_res2.rpca_apg, number = test_number) / test_number
# face_time_cost[2, 0] = timeit(face_rpca_res3.rpca_apg, number = test_number) / test_number
# face_time_cost[1, 0] = timeit(face_rpca_res4.rpca_apg, number = test_number) / test_number
# face_time_cost[0, 0] = timeit(face_rpca_res5.rpca_apg, number = test_number) / test_number


# face_time_cost[4, 1] = timeit(face_rpca_res1.rpca_ealm, number = test_number) / test_number
# face_time_cost[3, 1] = timeit(face_rpca_res2.rpca_ealm, number = test_number) / test_number
# face_time_cost[2, 1] = timeit(face_rpca_res3.rpca_ealm, number = test_number) / test_number
# face_time_cost[1, 1] = timeit(face_rpca_res4.rpca_ealm, number = test_number) / test_number
# face_time_cost[0, 1] = timeit(face_rpca_res5.rpca_ealm, number = test_number) / test_number

# face_time_cost[4, 2] = timeit(face_rpca_res1.rpca_ialm, number = test_number) / test_number
# face_time_cost[3, 2] = timeit(face_rpca_res2.rpca_ialm, number = test_number) / test_number
# face_time_cost[2, 2] = timeit(face_rpca_res3.rpca_ialm, number = test_number) / test_number
# face_time_cost[1, 2] = timeit(face_rpca_res4.rpca_ialm, number = test_number) / test_number
# face_time_cost[0, 2] = timeit(face_rpca_res5.rpca_ialm, number = test_number) / test_number



xaxis = [0.2, 0.4, 0.6, 0.8, 1]
# plt.figure(1)
# plt.plot(xaxis, demo_time_cost[:, 0], 'r', label = 'APG Method')
# plt.plot(xaxis, demo_time_cost[:, 1], 'g', label = 'EALM Method')
# plt.plot(xaxis, demo_time_cost[:, 2], 'b', label = 'IALM Method')
# plt.title('Speed Comparison at Five Resolutions for highway data')
# plt.legend()
# plt.xlabel('Scale')
# plt.ylabel('Time Cost(s)')

plt.figure(2)
plt.plot(xaxis[:4], esca_time_cost[:4, 0], 'r', label = 'APG Method')
plt.plot(xaxis[:4], esca_time_cost[:4, 1], 'g', label = 'EALM Method')
plt.plot(xaxis[:4], esca_time_cost[:4, 2], 'b', label = 'IALM Method')
plt.title('Speed Comparison at Five Resolutions for escalator data')
plt.legend()
plt.xlabel('Scale')
plt.ylabel('Time Cost(s)')

# plt.figure(3)
# plt.plot(xaxis, face_time_cost[:, 0], 'r', label = 'APG Method')
# plt.plot(xaxis, face_time_cost[:, 1], 'g', label = 'EALM Method')
# plt.plot(xaxis, face_time_cost[:, 2], 'b', label = 'IALM Method')
# plt.title('Speed Comparison at Five Resolutions for face specularity removal')
# plt.legend()
# plt.xlabel('Scale')
# plt.ylabel('Time Cost(s)')

plt.show()






