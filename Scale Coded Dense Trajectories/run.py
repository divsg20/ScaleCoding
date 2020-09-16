from Generate_Features import computeDenseTraj
import numpy as np
import Training_tools as tools
import pickle

def main(video_path, show_track=False, quality=0.001, min_distance=5, init_gap=1):

	"""
	:param video_path: path to video to generate scale coded features
	:param show_track: True/False display dense trajectories or not
	:param quality: experimental threshold value
	:param min_distance: grid spacing W = 5 pixels
	:return: scale coded dense trajectory features
	"""
	# dimension of generated dense trajectory features is 436
	features = np.zeros([0, 436], dtype=np.float32)

	#dt features will be written to (video_path)_features.txt
	dtFeatures = computeDenseTraj(video_path, show_track, quality, min_distance, init_gap)

	#get training cluster centers
	centers_traj = pickle.load(open('KTH_centers_traj.p', 'rb'))
	centers_hog = pickle.load(open('KTH_centers_hog.p', 'rb'))
	centers_hof = pickle.load(open('KTH_centers_hof.p', 'rb'))
	centers_mbhx = pickle.load(open('KTH_centers_mbhx.p', 'rb'))
	centers_mbhy = pickle.load(open('KTH_centers_mbhy.p', 'rb'))

	print('Generating bag-of-words for each video')
	# stack clusters for each descriptor to form 2_d array where each row corresponds to clusters for descriptor
	clusters = [centers_traj, centers_hog, centers_hof, centers_mbhx, centers_mbhy]

	#generate scale coded features
	SC_features = tools.get_spat_pyramids(dtFeatures, clusters)
	scpath = video_path[:-4] + "_features_scale_coded.txt"
	np.savetxt(SC_features, scpath)

if __name__ == "__main__":
	#change path to you video
	root = 'C:\\Users\\user 5\\Desktop\\Uni\\Masters\\Paper_Recreation\\py\\Datasets\\KTH\\boxing\\person01_boxing_d1_uncomp.avi'
	main(root)

