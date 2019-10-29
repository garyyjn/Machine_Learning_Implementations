import numpy as np
cimport numpy as np
cimport cython
from sklearn.cluster import KMeans

def dp_v1_python(a, n, K, dp, dpmean, dplast):
# It clusters each parameter for 'a'. To acclerating, it only enumerate "j" in a small range.
	for i in range(n):
		sx = 0
		sx2 = 0
		for j in range(i, -1, -1):
			if i - j > n / K * 2:
				break
			sx += a[j]
			sx2 += a[j] ** 2
			mean = sx / (i - j + 1)
			cost = sx2 - mean * sx
			for k in range(1, K + 1):
				if j > 0:
					t = dp[j - 1, k - 1]
				else:
					t = 0 if k == 1 else 1e100
				t += cost
				if dp[i, k] > t:
					dp[i, k] = t
					dpmean[i, k] = mean
					dplast[i, k] = j - 1

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef dp_v1_cython(np.ndarray[np.float32_t, ndim=1] a, int n, int K, np.ndarray[np.float32_t, ndim=2] dp, np.ndarray[np.float32_t, ndim=2] dpmean, np.ndarray[np.int32_t, ndim=2] dplast):
# Cython version of dp_v1_python
	cdef int i, j, k
	cdef int threshold = n / K * 2
	cdef float sx, sx2, mean, cost, t
	for i in range(n):
		sx = 0
		sx2 = 0
		for j in range(i, -1, -1):
			if i - j > threshold:
				break
			sx += a[j]
			sx2 += a[j] ** 2
			mean = sx / (i - j + 1)
			cost = sx2 - mean * sx
			for k in range(1, K + 1):
				if j > 0:
					t = dp[j - 1, k - 1]
				else:
					t = 0 if k == 1 else dp[i, k]
				t += cost
				if dp[i, k] > t:
					dp[i, k] = t
					dpmean[i, k] = mean
					dplast[i, k] = j - 1

def dp_v2_python(a, n, K, dp, dpmean, dplast):
# It clusters each parameter for a layer. 
	for i in range(n):
		sx = 0
		sx2 = 0
		for j in range(i, -1, -1):
			sx += a[j]
			sx2 += a[j] ** 2
			mean = sx / (i - j + 1)
			cost = sx2 - mean * sx
			for k in range(1, K + 1):
				if j > 0:
					t = dp[j - 1, k - 1]
				else:
					t = 0 if k == 1 else 1e100
				t += cost
				if dp[i, k] > t:
					dp[i, k] = t
					dpmean[i, k] = mean
					dplast[i, k] = j - 1

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef dp_v2_cython(np.ndarray[np.float32_t, ndim=1] a, int n, int K, np.ndarray[np.float32_t, ndim=2] dp, np.ndarray[np.float32_t, ndim=2] dpmean, np.ndarray[np.int32_t, ndim=2] dplast):
# Cython version of dp_v2_python
	cdef int i, j, k
	cdef float sx, sx2, mean, cost, t
	for i in range(n):
		sx = 0
		sx2 = 0
		for j in range(i, -1, -1):
			sx += a[j]
			sx2 += a[j] ** 2
			mean = sx / (i - j + 1)
			cost = sx2 - mean * sx
			for k in range(1, K + 1):
				if j > 0:
					t = dp[j - 1, k - 1]
				else:
					t = 0 if k == 1 else dp[i, k]
				t += cost
				if dp[i, k] > t:
					dp[i, k] = t
					dpmean[i, k] = mean
					dplast[i, k] = j - 1

def get_means(a, n, K, method='dp_v2_c'):
# method: ['dp_v1_c', 'dp_v1_python', 'dp_v2_c', 'dp_v2_python', 'kmeans_uniform', 'kmeans++']
	if 'dp' in method:
		dp = np.zeros((n, K + 1), dtype='float32') + np.inf
		dpmean = np.zeros((n, K + 1), dtype='float32')
		dplast = np.zeros((n, K + 1), dtype='int32') - 1
		if method == 'dp_v1_c':
			dp_v1_cython(a, n, K, dp, dpmean, dplast)
		elif method == 'dp_v2_c':
			dp_v2_cython(a, n, K, dp, dpmean, dplast)
		elif method == 'dp_v1_python':
			dp_v1_python(a, n, K, dp, dpmean, dplast)
		elif method == 'dp_v2_python':
			dp_v2_python(a, n, K, dp, dpmean, dplast)
		else:
			raise NotImplementedError('{} not implemented!\n'.format(method))
		j = n - 1
		k = K
		means = []
		while j >= 0:
			means.append(dpmean[j, k])
			j = dplast[j, k]
			k -= 1
		return means
	elif 'kmeans' in method:
		a = a.reshape(-1, 1)
		if method == 'kmeans++':
			kmeans = KMeans(n_clusters=K, random_state=1234, precompute_distances=True).fit(a)
		else:
			init = np.linspace(a[0], a[-1], num=K).reshape(-1, 1)
			kmeans = KMeans(n_clusters=K, init=init, n_init=1).fit(a)
		return kmeans.cluster_centers_.ravel().tolist()
