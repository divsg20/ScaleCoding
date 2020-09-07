import cv2
import numpy as np
from sklearn.neighbors import KDTree
import time
from scipy.cluster.vq import *
from numba import jit, cuda
"""
features: set of features
n_samples: number of samples to randomly select
returns: array with n_samples random samples from features
"""
def get_random_feature_samples(features, n_samples):
	n_features = features.shape[1]
	if n_features <= n_samples:
		return features
	index = np.arange(n_features)
	perm = np.random.permutation(index)
	sel_index = perm[:n_samples]

	# write to file
	"""
	out = open(output_file, 'w')
	for i in sel_index:
		line = features[i]
		out.write(line)
		out.write('\n')
	out.close()
	"""

	print('%d features extracted, %d of them are selected' % (n_features, n_samples))

	# return array with selected features
	mask = np.zeros_like(index, dtype=bool)
	mask[sel_index] = True
	return features[mask]


# generate bag of words given centers of visual words
def generateBOW(video_features, tree):
	if video_features.size == 0:
		return np.zeros(int(tree.sum_weight))
	#tree = KDTree(clusters)
	indexMatch = []
	for f in np.vstack(video_features):
		f = f.reshape(1, -1)
		_, index = tree.query(f, k=1)
		indexMatch.append(index[0, 0])
	indexMatch = np.array(indexMatch)
	hist, _ = np.histogram(indexMatch, int(tree.sum_weight))
	return hist

def cluster(clusterFeatures, k):
	attempts = 8
	print('Generating ' + str(k) + ' clusters')
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, attempts, 1.0)
	compactness, labels, centers = cv2.kmeans(np.float32(clusterFeatures), k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
	return compactness, labels, centers
