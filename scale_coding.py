import numpy as np
import fhog
import BoW
import cv2
from numba import jit
from operator import itemgetter
import time

# Features are a dictionary with the following structure
"""
features = {'frame': frame_num, 'scale': scale_level, 'width': bbox_width, 'height': bbox_height,
                                     'features': feature_array}
"""


def scalecode(features, SS, SM, clusters):

	cols = features[0]['features'].shape[1]
	words = int(clusters.sum_weight)

	Ss = np.zeros([0, cols], dtype=np.float32)
	Sm = np.zeros([0, cols], dtype=np.float32)
	Sl = np.zeros([0, cols], dtype=np.float32)

	Ss_1 = np.zeros([0, cols], dtype=np.float32)
	Ss_2 = np.zeros([0, cols], dtype=np.float32)
	Ss_3 = np.zeros([0, cols], dtype=np.float32)

	Sm_1 = np.zeros([0, cols], dtype=np.float32)
	Sm_2 = np.zeros([0, cols], dtype=np.float32)
	Sm_3 = np.zeros([0, cols], dtype=np.float32)

	Sl_1 = np.zeros([0, cols], dtype=np.float32)
	Sl_2 = np.zeros([0, cols], dtype=np.float32)
	Sl_3 = np.zeros([0, cols], dtype=np.float32)

	num_frames = max(map(itemgetter('frame'), features))

	fet = np.array(list(map(itemgetter('features'), features)))
	frames = np.array(list(map(itemgetter('frame'), features)))
	frames = frames/num_frames
	scales = np.array(list(map(itemgetter('scale'), features)))

	s_i = np.where(scales < SS)
	m_i = np.where(np.logical_and(scales >= SS, scales < SM))
	l_i = np.where(scales >= SM)
	t1 = np.where(frames <= 0.33)
	t2 = np.where(np.logical_and(frames > 0.33, frames <= 0.66))
	t3 = np.where(np.logical_and(frames > 0.66, frames <= 1.0))
	Ss = fet[s_i]
	Sl = fet[l_i]
	Sm = fet[m_i]

	Ss_1 = fet[np.intersect1d(s_i, t1)]
	Ss_2 = fet[np.intersect1d(s_i, t2)]
	Ss_3 = fet[np.intersect1d(s_i, t3)]

	Sm_1 = fet[np.intersect1d(m_i, t1)]
	Sm_2 = fet[np.intersect1d(m_i, t2)]
	Sm_3 = fet[np.intersect1d(m_i, t3)]

	Sl_1 = fet[np.intersect1d(l_i, t1)]
	Sl_2 = fet[np.intersect1d(l_i, t2)]
	Sl_3 = fet[np.intersect1d(l_i, t3)]

	# Without Paralelization
	H = np.hstack([BoW.generateBOW(Ss, clusters), BoW.generateBOW(Sm, clusters), BoW.generateBOW(Sl, clusters)])
	h_t1 = np.hstack([BoW.generateBOW(Ss_1, clusters), BoW.generateBOW(Sm_1, clusters), BoW.generateBOW(Sl_1, clusters)])
	h_t2 = np.hstack([BoW.generateBOW(Ss_2, clusters), BoW.generateBOW(Sm_2, clusters), BoW.generateBOW(Sl_2, clusters)])
	h_t3 = np.hstack([BoW.generateBOW(Ss_3, clusters), BoW.generateBOW(Sm_3, clusters), BoW.generateBOW(Sl_3, clusters)])
	Ht = h_t1 + h_t2 + h_t3

	#l1_norm = np.linalg.norm(Ht, 1)
	#if l1_norm != 0:
		#Ht = np.sqrt(np.divide(Ht, l1_norm))

	return H, Ht

# cell size for HOG
cell_size = 4
min_cells = 3   # minimum number of cells patch must be divided into
scale_num = 9   # max number of scales to extract
scaling_factor = np.sqrt(2)

# options - construct BoW for each scale or construct one BoW for all scales and compute BoW separately

# clusters - cluster centres of BoW vocbulary
def absolute_SC(features, clusters):
	scales = np.unique(np.array(list(map(itemgetter('scale'), features))))
	ss = scales[int(scales.shape[0] / 3)]
	sm = scales[int(scales.shape[0] / 3 * 2)]
	H, Ht = scalecode(features, ss, sm, clusters)
	return H, Ht


def relative_SC(features, clusters, w_, h_):
	for f in features:
		s_ = (f['width'] + f['height']) / (w_ + h_)
		f['scale'] = f['scale'] * s_

	scales = np.unique(np.array(list(map(itemgetter('scale'), features))))
	St_ = scales.shape[0]  # cardinality normalising factor
	ss = scales[int(St_ / 3)]
	sm = scales[int(St_ / 3 * 2)]

	H, Ht = scalecode(features, ss, sm, clusters)

	H = np.divide(H, St_)
	Ht = np.divide(Ht, St_)
	return H, Ht

"""
funtion creating array of frames at specififed scales
frame = video frame
sizes = array of dimensions (width, height) of frames at each scale. 1D array of tuples with 8 elements max
type = image type
returns
scaled_frames: list of same image frames at diff scales
"""


def get_scaled_frames(frame, scales):
	scaled_frames = [frame]
	for s in scales:
		#print(s)
		if s == 1:
			continue
		try:
			a = cv2.resize(frame, (int(np.ceil(frame.shape[1] / s)), int(np.ceil(frame.shape[0] / s))), interpolation=cv2.INTER_LINEAR)
		except:
			return -1
		scaled_frames.append(a)
	return scaled_frames

def get_scales(bbox):
	min_size = min(bbox[2], bbox[3])
	scale = 1
	scales = []
	nlayers = 0
	while (min_size // cell_size) >= min_cells and nlayers <= scale_num:
		scales.append(scale)
		scale = scale * scaling_factor
		min_size = min_size / scaling_factor

		nlayers = nlayers + 1
	print(scales)
	return scales
