import cv2
import numpy as np

scale_num = 8
scale_stride = np.sqrt(2)
track_len = 15 				# max number of frames for which trajectories are computed

# Descriptor parameters
patch_size = 32
nxy_cell = 2
nt_cell = 3
min_flow = 0.4
epsilon = 0.05

# parameters for rejecting trajectory
min_var = np.sqrt(3)
max_var = 50.
max_dis = 20.
"""
funtion finding number of scales video will be tracked at. Max is 8.
frame = single frame of to-be processed vid
returns:
scales = array of scales for vid to be processed at. 1D float array wih 8 elements max
sizes = array of dimensions (width, height) of frames at each scale. 1D array of tuples with 8 elements max
"""


def init_pry(cap):
	rows = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	cols = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	min_size = min(rows, cols)

	nlayers = 0
	while min_size >= patch_size and nlayers <= scale_num:
		min_size = min_size / scale_stride
		nlayers = nlayers + 1

	# at least 1 scale
	if nlayers is 0:
		nlayers = 1

	scales = np.ones(nlayers)
	sizes = np.empty((nlayers, 2), dtype=int)
	sizes[:, :] = (cols, rows)

	# populate scales and sizes with info needed for multi-scale feature extraction
	for i in range(1, nlayers):
		scales[i] = scales[i-1] * scale_stride
		sizes[i, :] = np.round(np.true_divide(sizes[i], scales[i]))
	return scales, sizes


"""
funtion creating array of frames at specififed scales
frame = video frame
sizes = array of dimensions (width, height) of frames at each scale. 1D array of tuples with 8 elements max
type = image type
returns
scaled_frames: list of same image frames at diff scales
"""


def build_pry(frame, sizes):
	nlayers = sizes.shape[0]
	scaled_frames = [frame]
	for i in range(1, nlayers):
		a = cv2.resize(frame, (sizes[i][0], sizes[i][1]), interpolation=cv2.INTER_LINEAR)
		scaled_frames.append(a)
	return scaled_frames


"""
funtion dense sampling an image
img = video frame image to be densely sampled MUST BE GREYSCALE
points = array of points(x, y) currently being tracked
grid_spacing = int value specifying sampling step size
quality = set value specifying compromise between saliency and density of the sampled points
returns
dense_points: array of densely sampled points
"""


def dense_sample(img, points, grid_spacing, quality):
	grid_width = img.shape[1] // grid_spacing
	grid_height = img.shape[0] // grid_spacing
	x_max = grid_width * grid_spacing
	y_max = grid_height * grid_spacing

	eig = cv2.cornerMinEigenVal(img, 3, 3)
	max_val = np.amax(eig)		# can use minmaxloc
	T = quality * max_val		# threshold
	# ensure dense coverage verify presence of tracking point at every frame -----IS THIS NEEDED???
	counter = np.zeros(grid_width * grid_height)
	for i in points:
		x = np.floor(i[0])
		y = np.floor(i[1])
		if x >= x_max or y >= y_max:
			continue
		x = x // grid_spacing
		y = y // grid_spacing
		index = y * grid_width + x
		counter[int(index)] = counter[int(index)] + 1

	# get densely sampled points
	index = 0
	offset = grid_spacing // 2
	dense_points = []
	for i in range(grid_height):
		for j in range(grid_width):
			if counter[index] > 0:
				index = index + 1
				continue
			x = (j * grid_spacing) + offset		# x is width is cols
			y = (i * grid_spacing) + offset		# y is height is rows
			if eig[y][x] > T:
				dense_points.append([x, y])
			index = index + 1
	dense_points_np = np.array(dense_points)
	return dense_points_np


"""
Computed dense optical flow, applying M x M median filter kernel for all scales
grey_frame, prev_grey_frame = consecutive frames at original scale
scales = actual scale levels computer by initpry
sizes = array of computed dimensions for each scale which vid will be sampled at (determined by init_pry)
M = median kernal size
flow = smoothed optical flow at each sampled scale. Opt flow for each scale has 2D for (x, y) points
"""


def calcOptFlow(grey_frame, prev_grey_frame, scales, sizes,  m):
	prev_frames = build_pry(prev_grey_frame, sizes)
	nxt_frames = build_pry(grey_frame, sizes)
	# flow = [cv2.calcOpticalFlowFarneback(prev_frames[0], nxt_frames[0], None, 0.5, 1, 10, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)]
	flow = []
	length = len(sizes)
	for i in range(length):
		"""
		    pyr_scale = 1/sqrt(2) as that is the chosen scale_stride
		    levels = 1 (number of pyramid layers including the initial image)
		    winsize = 10
		    iterations = 2
		    poly_n = 7(size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values is more robust)
		    poly_sigma = 1.5
		    flow = matrx of size = to image. Shape [height, width, 2] (2 channels for (x,y))
		"""
		"""
		sigma = (scales[i] - 1) * 0.5
		round_sigma = int(np.round(sigma * 5)) | 1
		smooth_sz = max(round_sigma, 3)
		p = cv2.GaussianBlur(np.float32(prev_frames[i]), (smooth_sz, smooth_sz), sigma, None, sigma)
		print(p)
		n = cv2.GaussianBlur(np.float32(nxt_frames[i]), (smooth_sz, smooth_sz), sigma, None, sigma)
		"""
		f = cv2.calcOpticalFlowFarneback(prev_frames[i], nxt_frames[i], None, 1/scale_stride, 1, 10, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
		# median filter kernel M x M
		flow.append(cv2.medianBlur(f, m))
	return flow

"""
Object Holding Cell information for descriptor computation
"""
class RectInfo:
	__slots__ =['x', 'y', 'height', 'width']

	def __init__(self, pnt, img_height, img_width, descinfo):
		x_min = descinfo.width / 2
		y_min = descinfo.height / 2
		x_max = img_width - descinfo.width
		y_max = img_height - descinfo.height

		self.x = int(min(max(np.round(pnt[0] - x_min), 0), x_max))
		self.y = int(min(max(np.round(pnt[1] - y_min), 0), y_max))

		self.height = descinfo.height
		self.width = descinfo.width


"""
Object holding all information about descriptors
"""


class DescInfo:
	__slots__ = ['nBins',  	# number of bins for vector quanitization
                 'isHof',
                 'nxCells',  # number of cells in x direction
                 'nyCells',  # number of cells in y direction
                 'ntCells',
                 'dim',  	# dimension of the descriptor
                 'height',  # size of the block fr computing the descriptor
                 'width']

	def __init__(self, nbins, ishof, height, width, nxycell, ntcell):
		self.nBins = nbins
		self.isHof = ishof
		self.nxCells = nxycell
		self.nyCells = nxycell
		self.ntCells = ntcell
		self.dim = nbins * nxycell * nxycell
		self.height = height
		self.width = width
		assert type(self.nBins) == int, "bin_size should be integer,"
		assert type(self.nxCells) == int, "cell_size should be integer,"
		assert type(self.nyCells) == int, "cell_size should be integer,"
		assert type(self.ntCells) == int, "cell_size should be integer,"


"""
Integral Histogram calculation to be later processed to obtain descriptors
desc is a float array with histogram computed for each point stored
"""
def buildDescMat(gx, gy, descinfo):
	nbin = descinfo.nBins - 1 if descinfo.isHof else descinfo.nBins
	ndims = descinfo.nBins
	anglebase = nbin/360.0
	index = 0
	# Python Calculate gradient magnitude and direction ( in degrees )
	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
	desc = np.zeros([gx.shape[0] + 1, gx.shape[1] + 1, ndims])
	for i in range(gx.shape[0]):			# going through rows
		for j in range(gx.shape[1]):		# going through columns
			# ensure not invalid(nan or inf)
			mag[i, j] = np.nan_to_num(mag[i, j])
			# for zero bin of HOF
			if descinfo.isHof and (mag[i, j] <= min_flow):
				bin0 = nbin				# zero bin is the last one, index 8
				mag0 = 1.0
				bin1 = 0
				mag1 = 0
			else:
				fbin = anglebase * (angle[i, j] % 360.)
				bin0 = int(np.floor(fbin))
				bin1 = int((bin0 + 1) % nbin)
				mag1 = (fbin - bin0) * mag[i, j]
				mag0 = mag[i, j] - mag1
				if np.isnan(mag0) or np.isnan(mag1):
					print(i, j)
			desc[i][j][bin0] = mag0
			desc[i][j][bin1] = mag1
	desc = cv2.integral(desc)
	return desc

"""
Get descriptor from integral histogram
"""


def getDesc(int_hist, rectinfo, descinfo, img_height, img_width):
	dim = descinfo.dim
	nBins = descinfo.nBins
	xStride = int(rectinfo.width / descinfo.nxCells)
	yStride = int(rectinfo.height / descinfo.nyCells)

	# iterate over different cells
	iDesc = 0
	vec = np.empty(dim)
	xPos = int(rectinfo.x)
	for x in range(descinfo.nxCells):
		yPos = int(rectinfo.y)
		for y in range(descinfo.nyCells):
			# positions in the integral histogram

			
			# to get hist it's: top_left + bottom_right - bottom_left - top_right
			for i in range(nBins):
				top_left = int_hist[yPos, xPos, i]
				top_right = int_hist[yPos, xPos + xStride, i]
				bottom_left = int_hist[yPos + yStride, xPos, i]
				bottom_right = int_hist[yPos + yStride, xPos + xStride, i]
				sum_hists = top_left + bottom_right - bottom_left - top_right
				vec[iDesc] = max(sum_hists, 0) + epsilon
				iDesc = iDesc + 1
			yPos = yPos + yStride
		xPos = xPos + xStride
	# normalise
	#RootSIFT norm
	l1_norm = np.linalg.norm(vec, 1)
	hist = np.sqrt(np.divide(vec, l1_norm))

	return hist
"""
Descriptor Computation for all scales
"""


def hogComp(img, descinfo):
	# calculate gradients
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
	h = buildDescMat(gx, gy, descinfo)
	return h


def hofComp(flow, descinfo):
	gx, gy = cv2.split(flow)
	h = buildDescMat(gx, gy, descinfo)
	return h


def mbhComp(flow, descinfo):
	flow_x, flow_y = cv2.split(flow)
	flow_xdx = cv2.Sobel(flow_x, cv2.CV_32F, 1, 0, ksize=1)
	flow_xdy = cv2.Sobel(flow_x, cv2.CV_32F, 0, 1, ksize=1)
	flow_ydx = cv2.Sobel(flow_y, cv2.CV_32F, 1, 0, ksize=1)
	flow_ydy = cv2.Sobel(flow_y, cv2.CV_32F, 0, 1, ksize=1)

	mbhx = buildDescMat(flow_xdx, flow_xdy, descinfo)
	mbhy = buildDescMat(flow_ydx, flow_ydy, descinfo)
	return mbhx, mbhy

"""
Computation of trajectory shape. Returns true if he trajectory is valid and untrue if invalid
"""

def tShapeComp(t, length):
	# check for static/ random trajectories
	mean_x = np.mean(t[:, 0])
	mean_y = np.mean(t[:, 1])
	var_x = np.sqrt(np.sum(np.square(np.subtract(t[:, 0], mean_x))) / t.shape[0])
	var_y = np.sqrt(np.sum(np.square(np.subtract(t[:, 1], mean_y))) / t.shape[0])

	# remove static trajectory
	if (var_x < min_var) and (var_y < min_var):
		return False, [], 0, 0, 0, 0, 0
	# remove random trajectory
	if (var_x > max_var) or (var_y > max_var):
		return False, [], 0, 0, 0, 0, 0

	S = np.empty((length, 2))
	lengths = 0
	cur_max = 0
	# compute displacement vectors
	for i in range(length):
		S[i] = t[i + 1] - t[i]
		lengths = lengths + np.linalg.norm(S[i])
		cur_max = max(lengths, cur_max)
	# check  if the displacement vector s is larger than 70% of the overall displacement of the trajectory
	if (cur_max > max_dis) and (cur_max > length * 0.7):
		return False, [], 0, 0, 0, 0, 0

	# normalise trajectory
	S = S / lengths
	return True, S, mean_x, mean_y, var_x, var_y, lengths


"""
Print descriptor to file for storage
"""
def getFeature(descinfo, descmat, length):
	tStride = int(np.round(length / descinfo.ntCells))
	dim = descinfo.dim
	pos = 0
	vec = np.zeros(dim)
	# each row has histogram
	feature = np.zeros([dim * descinfo.ntCells], dtype=np.float32)
	# all_vec = np.empty((dim, descinfo.ntCells))
	for i in range(descinfo.ntCells):
		vec.fill(0)
		for t in range(tStride):
			for j in range(dim):
				vec[j] = vec[j] + descmat[pos]
				pos = pos + 1
		vec = vec / tStride
		feature[i * dim:(i * dim) + dim] = vec
	return feature


def printDesc(f_writer, descinfo, descmat, length):
	tStride = int(np.round(length / descinfo.ntCells))
	dim = descinfo.dim
	pos = 0
	vec = np.zeros(dim)
	# all_vec = np.empty((dim, descinfo.ntCells))
	for i in range(descinfo.ntCells):
		vec.fill(0)
		for t in range(tStride):
			for j in range(dim):
				vec[j] = vec[j] + descmat[pos]
				pos = pos + 1
		vec = vec / tStride
		f_writer.writerow(vec)
		# all_vec[:][i] = vec

def printDesc_txt(f_writer, descinfo, descmat, length):
	tStride = int(np.round(length / descinfo.ntCells))
	dim = descinfo.dim
	pos = 0
	vec = np.zeros(dim)
	# all_vec = np.empty((dim, descinfo.ntCells))
	for i in range(descinfo.ntCells):
		vec.fill(0)
		for t in range(tStride):
			for j in range(dim):
				vec[j] = vec[j] + descmat[pos]
				pos = pos + 1
		vec = vec / tStride
		for j in range(dim):
			f_writer.write("%.7f\t" % vec[j])
		# all_vec[:][i] = vec
"""
draw out trajectoreis for visualisation
"""


def drawTrack(points, index, scale, img):
	# obtain the points up to and including indicated index and scale them
	points_ = np.multiply(points[0:index + 1], scale)
	for i in range(index):
		cv2.line(img, (int(points_[i][0]), int(points_[i][1])), (int(points_[i + 1][0]), int(points_[i + 1][1])),
				 (0, int(np.round(255 * (i + 1) / (index + 1))), 0), 2)
	cv2.circle(img, (int(points_[0][0]), int(points_[0][1])), 2, (0, 0, 255), -1)



