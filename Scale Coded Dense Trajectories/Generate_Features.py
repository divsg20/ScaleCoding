"""
Computes all descriptors for dense trajectories and writes them to a a txt file
"""

import sys
import os
import cv2
import numpy as np
import Descriptors as desc
# from OpticalFlow import save_3D, save_desc, save_desc_form, show_opt_flow
from DenseTrackStab import Track, Trackerinit


def computeDenseTraj(path, show_track, quality, min_distance, init_gap):
	# show_track = False
	# quality = 0.001             # experimental threshold value
	# min_distance = 5            # grid spacing of W = 5 pixels
	frame_count = 0

	hogInfo = desc.DescInfo(8, False, desc.patch_size, desc.patch_size, desc.nxy_cell, desc.nt_cell)
	hofInfo = desc.DescInfo(9, True, desc.patch_size, desc.patch_size, desc.nxy_cell, desc.nt_cell)
	mbhInfo = desc.DescInfo(8, False, desc.patch_size, desc.patch_size, desc.nxy_cell, desc.nt_cell)

	tracks = []  # list of sampled feature points at each scale
	cap = cv2.VideoCapture(path)
	if cap.isOpened():
		vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		vid_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		fps = cap.get(cv2.CAP_PROP_FPS)
		print('width: %f \n height: %f \n fps: %f \n frames: %f \n' % (vid_width, vid_height, fps, vid_length))
		scales, sizes = desc.init_pry(cap)
	# hog = np.zeros([hogInfo.dim * desc.nt_cell, scales.shape[0]], dtype=np.float32)
	else:
		sys.exit("Unable to open the video: " + path)
	# file to write feature info into
	txtFile = open(path[:-4] + "_features.txt", "w")
	# alocate memory for features
	features = np.empty([0, 436])
	# writer = csv.writer(csvFile)

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("could not get decode video")
			break
		grey = cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2GRAY)
		scaled_frames = desc.build_pry(grey, sizes)  # gets frame at multiple spacial scales
		if frame_count == 0:  # start frame
			# create tracker objects
			for iScale in range(len(scaled_frames)):
				points = []
				points = desc.dense_sample(scaled_frames[iScale], points, min_distance, quality)
				# save points in tracker object for each point on given iScale
				tracks.append(Trackerinit(points, desc.track_len, hogInfo, hofInfo, mbhInfo))
				del points
			prev_grey = grey
			prev_scaled_frames = scaled_frames
			frame_count = frame_count + 1
			continue

		# get optical flow for all scales. Function accounts for this
		flow = desc.calcOptFlow(grey, prev_grey, scales, sizes, 5)
		for iScale in range(len(scaled_frames)):
			width = scaled_frames[iScale].shape[1]
			height = scaled_frames[iScale].shape[0]

			# compute integral histograms for descriptors for all scales
			hog_int_hist = desc.hogComp(prev_scaled_frames[iScale], hogInfo)
			hof_int_hist = desc.hofComp(flow[iScale], hofInfo)
			mbhx_int_hist, mbhy_int_hist = desc.mbhComp(flow[iScale], mbhInfo)

			# track feature points at each scale
			i = 0
			points_ = []
			# to_delete = []

			for iTrack in tracks[iScale]:
				index = iTrack.index
				prev_point = iTrack.points[index]
				x = int(min(max(np.round(prev_point[0]), 0), width - 1))
				y = int(min(max(np.round(prev_point[1]), 0), height - 1))
				point = np.array([prev_point[0] + flow[iScale][y][x][0], prev_point[1] + flow[iScale][y][x][1]])

				# store point for dense sampling
				points_.append(point)
				# if point out of bounds, store index to remove
				if point[0] <= 0 or point[0] >= width or point[1] <= 0 or point[1] >= height:
					tracks[iScale] = np.delete(tracks[iScale], i)
					continue
				# get descriptors for each feature point from integral histogram
				rect = desc.RectInfo(prev_point, height, width, hogInfo)
				start_index = index * hogInfo.dim
				end_index = start_index + hogInfo.dim
				hof_start_index = index * hofInfo.dim
				iTrack.hog[start_index: end_index] = desc.getDesc(hog_int_hist, rect, hogInfo, height, width)
				# hof has different dimension to rest
				iTrack.hof[hof_start_index: hof_start_index + hofInfo.dim] = desc.getDesc(hof_int_hist, rect, hofInfo,
				                                                                          height, width)
				iTrack.mbhX[start_index: end_index] = desc.getDesc(mbhx_int_hist, rect, mbhInfo, height, width)
				iTrack.mbhY[start_index: end_index] = desc.getDesc(mbhy_int_hist, rect, mbhInfo, height, width)
				iTrack.addPoint(point)

				# draw trajectories
				if show_track and iScale == 0:
					desc.drawTrack(iTrack.points, iTrack.index, scales[iScale], frame)

				# Check if trajectory surpasses max track length
				if iTrack.index >= desc.track_len:
					# get points. Have one point more than max length for calculation of trajectory shape.
					# trajectories = iTrack.points[: desc.track_len + 1] * scales[iScale]

					isValid, Traj, mean_x, mean_y, var_x, var_y, traj_length = \
						desc.tShapeComp(iTrack.points * scales[iScale], desc.track_len)
					if isValid:
						f = np.empty([436], dtype=np.float32)
						# writer.writerow(['frame', 'x mean', 'y mean', 'x var', 'y var', 'length', 'scale'])
						txtFile.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t" %
						              (frame_count, mean_x, mean_y, var_x, var_y, traj_length, scales[iScale]))
						f[:7] = np.array([frame_count, mean_x, mean_y, var_x, var_y, traj_length, scales[iScale]])

						# for spatio temporal data
						# spatio width
						txtFile.write("%f\t" % min(max(mean_x / vid_width, 0.0), 0.999))
						# spatio_height
						txtFile.write("%f\t" % min(max(mean_y / vid_height, 0.0), 0.999))
						# spatio length
						txtFile.write("%f\t" % min(max((frame_count - desc.track_len / 2.0) / vid_length, 0.0), 0.999))
						# for spatio temporal data
						# spatio width
						f[7] = min(max(mean_x / vid_width, 0.0), 0.999)
						# spatio_height
						f[8] = min(max(mean_y / vid_height, 0.0), 0.999)
						# spatio length
						f[9] = min(max((frame_count - desc.track_len / 2.0) / vid_length, 0.0), 0.999)
						count = 0
						for t in Traj:
							txtFile.write("%f\t%f\t" % (t[0], t[1]))
							f[10 + count] = t[0]
							f[10 + count + 1] = t[1]
							count = count + 2

						# store descriptors
						# writer.writerow(["HOG"])
						desc.printDesc_txt(txtFile, hogInfo, iTrack.hog, desc.track_len)
						# writer.writerow(["HOF"])
						desc.printDesc_txt(txtFile, hofInfo, iTrack.hof, desc.track_len)
						# writer.writerow(["MBHX"])
						desc.printDesc_txt(txtFile, mbhInfo, iTrack.mbhX, desc.track_len)
						# writer.writerow(["MBHY"])
						desc.printDesc_txt(txtFile, mbhInfo, iTrack.mbhY, desc.track_len)

						# writer.writerow(["HOG"])
						f[40:136] = desc.getFeature(hogInfo, iTrack.hog, desc.track_len)
						# writer.writerow(["HOF"])
						f[136:244] = desc.getFeature(hofInfo, iTrack.hof, desc.track_len)
						# writer.writerow(["MBHX"])
						f[244:340] = desc.getFeature(mbhInfo, iTrack.mbhX, desc.track_len)
						# writer.writerow(["MBHY"])
						f[340: 436] = desc.getFeature(mbhInfo, iTrack.mbhY, desc.track_len)
						# writer.writerow([])
						features = np.vstack((features, f))
						del f
						# writer.writerow([])
						txtFile.write("\n")

					tracks[iScale] = np.delete(tracks[iScale], i)
					# to_delete.append(i)
					continue
				i = i + 1
			# remove unwanted points
			# tracks[iScale] = np.delete(tracks[iScale], to_delete)
			# free up memory
			del hog_int_hist
			del hof_int_hist
			del mbhx_int_hist
			del mbhy_int_hist

			# densely resample points every init_gap frames

			if frame_count % init_gap != 0:
				continue

			points_ = desc.dense_sample(scaled_frames[iScale], points_, min_distance, quality)
			# add as points to track
			tracks[iScale] = np.append(tracks[iScale], Trackerinit(points_, desc.track_len, hogInfo, hofInfo, mbhInfo),
			                           axis=0)
		# store previous frame
		prev_grey = grey
		prev_scaled_frames = scaled_frames
		frame_count = frame_count + 1

		if show_track:
			cv2.imshow('Dense Track', frame)
			if cv2.waitKey(0) & 0xFF == ord('q'):
				break
	if show_track:
		cv2.destroyAllWindows()
	cap.release()
	return features

