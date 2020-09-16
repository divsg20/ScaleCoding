import numpy as np
from sklearn.neighbors import KDTree
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import scipy.stats as stats
import time

num_words = 4000


"""
features: set of features
n_samples: number of samples to randomly select
returns: array with n_samples random samples from features
"""


def get_random_feature_samples(features, n_samples):
	n_features = len(features)
	assert (n_features >= n_samples)
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
def generateBOW(video_features, clusterCenters):
	tree = KDTree(clusterCenters)
	video_features = np.nan_to_num(video_features)
	indexMatch = []
	for f in video_features:
		f = f.reshape(1, -1)
		try:
			dist, index = tree.query(f, k=1)
		except:
			f = f[:clusterCenters.shape[1]]
			dist, index = tree.query(f, k=1)
		indexMatch.append(index[0, 0])
	indexMatch = np.array(indexMatch)
	hist, _= np.histogram(indexMatch, clusterCenters.shape[0])
	return hist

def get_spat_pyramids(features, clusters):
	pyr_1 = np.zeros([len(clusters), num_words])
	pyr_2 = np.zeros([len(clusters), num_words])
	pyr_3 = np.zeros([len(clusters), num_words])
	pyr_4 = np.zeros([len(clusters), num_words])
	pyr_5 = np.zeros([len(clusters), num_words])
	pyr_6 = np.zeros([len(clusters), num_words])

	pyr_1_S = []
	pyr_2_S = []
	pyr_3_S = []
	pyr_4_S = []
	pyr_5_S = []
	pyr_6_S = []

	# columns to get different descriptors
	# n is start col, n + 1 is end col for n [0, 5]
	desc_cols = np.array([10, 40, 136, 244, 340, 435])
	start = time.time()
	# Scale Code
	scal = features[:, 6]
	scales = np.unique(scal)
	print(scales)

	# dynamic setting of paramters
	sm = scales[int(scales.shape[0] / 3 * 2)]
	ss = scales[int(scales.shape[0] / 3 )]
	features = np.nan_to_num(features)

	Ss = features[features[:, 6] < ss, :]
	Sl = features[features[:, 6] >= sm, :]
	Sm = features[np.logical_and(features[:, 6] >= ss, features[:, 6] < sm), :]


	scale_coded = [Ss, Sm, Sl]
	for video_features in scale_coded:
		# pyr_1: h1 x v1 x t1 (whole video)

		# pyr_2: h3×v1×t1
		h3_1 = video_features[video_features[:, 8] <= 0.333, :]  # 9th column has spat height
		h3_2 = video_features[(video_features[:, 8] > 0.333) & (video_features[:, 8] <= 0.666),
		       :]  # 8 column has spat height
		h3_3 = video_features[video_features[:, 8] > 0.666, :]  # 8 column has spat height

		# pyr_3: h2×v2×t1
		h2_1_v2_1 = video_features[(video_features[:, 7] <= 0.4995) & (video_features[:, 8] <= 0.4995),
		            :]  # 7 col is spat width
		h2_2_v2_2 = video_features[(video_features[:, 7] > 0.4995) & (video_features[:, 8] > 0.4995), :]
		h2_2_v2_1 = video_features[(video_features[:, 7] > 0.4995) & (video_features[:, 8] <= 0.4995), :]
		h2_1_v2_2 = video_features[(video_features[:, 7] <= 0.4995) & (video_features[:, 8] > 0.4995), :]

		# pyr_4: h1×v1×t2
		t2_1 = video_features[video_features[:, 9] <= 0.4995, :]  # 9 col is temporal val
		t2_2 = video_features[video_features[:, 9] > 0.4995, :]  # 9 col is temporal val

		# pyr_5: h3×v1×t2
		h3_1_t2_1 = h3_1[h3_1[:, 9] <= 0.4995, :]
		h3_1_t2_2 = h3_1[h3_1[:, 9] > 0.4995, :]
		h3_2_t2_1 = h3_2[h3_2[:, 9] <= 0.4995, :]
		h3_2_t2_2 = h3_2[h3_2[:, 9] > 0.4995, :]
		h3_3_t2_1 = h3_3[h3_3[:, 9] <= 0.4995, :]
		h3_3_t2_2 = h3_3[h3_3[:, 9] > 0.4995, :]

		# pyr_6: h2×v2×t2
		h2_1_v2_1_t2_1 = h2_1_v2_1[h2_1_v2_1[:, 9] <= 0.4995, :]
		h2_2_v2_2_t2_1 = h2_2_v2_2[h2_2_v2_2[:, 9] <= 0.4995, :]
		h2_2_v2_1_t2_1 = h2_2_v2_1[h2_2_v2_1[:, 9] <= 0.4995, :]
		h2_1_v2_2_t2_1 = h2_1_v2_2[h2_1_v2_2[:, 9] <= 0.4995, :]
		h2_1_v2_1_t2_2 = h2_1_v2_1[h2_1_v2_1[:, 9] > 0.4995, :]
		h2_2_v2_2_t2_2 = h2_2_v2_2[h2_2_v2_2[:, 9] > 0.4995, :]
		h2_2_v2_1_t2_2 = h2_2_v2_1[h2_2_v2_1[:, 9] > 0.4995, :]
		h2_1_v2_2_t2_2 = h2_1_v2_2[h2_1_v2_2[:, 9] > 0.4995, :]
		for i in range(len(clusters)):
			# pyr_1: h1 x v1 x t1 (whole video)
			pyr_1[i] = generateBOW(video_features[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])
			# ROOT SIFT normalise
			# l1_norm = np.linalg.norm(pyr_1[i], 1)
			# if l1_norm != 0:
			# pyr_1[i] = np.sqrt(np.divide(pyr_1[i], l1_norm))

			# pyr_2: h3×v1×t1
			pyr_2[i] = generateBOW(h3_1[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])
			pyr_2[i] = np.sum([pyr_2[i], generateBOW(h3_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])], axis=0)
			pyr_2[i] = np.sum([pyr_2[i], generateBOW(h3_3[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])], axis=0)
			# ROOT SIFT normalise
			# l1_norm = np.linalg.norm(pyr_2[i], 1)
			# if l1_norm != 0:
			# pyr_2[i] = np.sqrt(np.divide(pyr_2[i], l1_norm))

			# pyr_3: h2×v2×t1
			pyr_3[i] = generateBOW(h2_1_v2_1[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])
			pyr_3[i] = np.sum([pyr_3[i], generateBOW(h2_2_v2_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_3[i] = np.sum([pyr_3[i], generateBOW(h2_2_v2_1[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_3[i] = np.sum([pyr_3[i], generateBOW(h2_1_v2_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			# ROOT SIFT normalise
			# l1_norm = np.linalg.norm(pyr_3[i], 1)
			# if l1_norm != 0:
			# pyr_3[i] = np.sqrt(np.divide(pyr_3[i], l1_norm))

			# pyr_4: h1×v1×t2
			pyr_4[i] = generateBOW(t2_1[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])
			pyr_4[i] = np.sum([pyr_4[i], generateBOW(t2_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])], axis=0)
			# ROOT SIFT normalise
			# l1_norm = np.linalg.norm(pyr_4[i], 1)
			# if l1_norm != 0:
			# pyr_4[i] = np.sqrt(np.divide(pyr_4[i], l1_norm))

			# pyr_5: h3×v1×t2
			pyr_5[i] = generateBOW(h3_1_t2_1[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])
			pyr_5[i] = np.sum([pyr_5[i], generateBOW(h3_1_t2_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_5[i] = np.sum([pyr_5[i], generateBOW(h3_2_t2_1[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_5[i] = np.sum([pyr_5[i], generateBOW(h3_2_t2_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_5[i] = np.sum([pyr_5[i], generateBOW(h3_3_t2_1[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_5[i] = np.sum([pyr_5[i], generateBOW(h3_3_t2_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			# ROOT SIFT normalise
			# l1_norm = np.linalg.norm(pyr_5[i], 1)
			# if l1_norm != 0:
			# pyr_5[i] = np.sqrt(np.divide(pyr_5[i], l1_norm))

			# pyr_6: h2×v2×t2
			pyr_6[i] = generateBOW(h2_1_v2_1_t2_1[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])
			pyr_6[i] = np.sum([pyr_6[i], generateBOW(h2_2_v2_2_t2_1[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_6[i] = np.sum([pyr_6[i], generateBOW(h2_2_v2_1_t2_1[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_6[i] = np.sum([pyr_6[i], generateBOW(h2_1_v2_2_t2_1[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_6[i] = np.sum([pyr_6[i], generateBOW(h2_1_v2_1_t2_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_6[i] = np.sum([pyr_6[i], generateBOW(h2_2_v2_2_t2_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_6[i] = np.sum([pyr_6[i], generateBOW(h2_2_v2_1_t2_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_6[i] = np.sum([pyr_6[i], generateBOW(h2_2_v2_2_t2_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)
			pyr_6[i] = np.sum([pyr_6[i], generateBOW(h2_1_v2_2_t2_2[:, desc_cols[i]: desc_cols[i + 1]], clusters[i])],
			                  axis=0)

		# ROOT SIFT normalise
		# l1_norm = np.linalg.norm(pyr_6[i], 1)
		# if l1_norm != 0:
		# pyr_6[i] = np.sqrt(np.divide(pyr_6[i], l1_norm))
		pyr_1_S.append(pyr_1)
		pyr_2_S.append(pyr_2)
		pyr_3_S.append(pyr_3)
		pyr_4_S.append(pyr_4)
		pyr_5_S.append(pyr_5)
		pyr_6_S.append(pyr_6)

	pyr_1_S = np.hstack(pyr_1_S)
	pyr_2_S = np.hstack(pyr_2_S)
	pyr_3_S = np.hstack(pyr_3_S)
	pyr_4_S = np.hstack(pyr_4_S)
	pyr_5_S = np.hstack(pyr_5_S)
	pyr_6_S = np.hstack(pyr_6_S)
	return np.hstack([pyr_1_S, pyr_2_S, pyr_3_S, pyr_4_S, pyr_5_S, pyr_6_S])