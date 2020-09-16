import cv2
import numpy as np
from Descriptors import DescInfo


"""
Takes in array of points, creates tracker object for each point and returns array of tracker oject
"""


def Trackerinit(points, length, hoginfo, hofinfo, mbhinfo):
    all_tracks = np.empty(points.shape[0], dtype=object)
    for i in range(points.shape[0]):
        dense_tracker = Track(length, points[i], hoginfo, hofinfo, mbhinfo)
        all_tracks[i] = dense_tracker
    return all_tracks


# Holds all information for tracking on a particular scale up to a certain number of frames
class Track:
    def __init__(self, length, pnt, hoginfo, hofinfo, mbhinfo):
        assert type(hoginfo) == DescInfo, "Expected type DescInfo for HOG"
        assert type(hofinfo) == DescInfo, "Expected type DescInfo for HOF"
        assert type(mbhinfo) == DescInfo, "Expected type DescInfo for MBH"
        # array of points being tracked on a given scale. Shape = [img_height, img_width, 2]
        self.points = np.empty((length + 1, 2))
        self.points[0] = pnt
        # descriptor information for tracked points on a given scales
        self.hog = np.empty(hoginfo.dim * length)
        self.hof = np.empty(hofinfo.dim * length)
        self.mbhX = np.empty(mbhinfo.dim * length)
        self.mbhY = np.empty(mbhinfo.dim * length)
        self.index = 0

    def addPoint(self, pnt):
        self.index = self.index + 1
        self.points[self.index] = pnt






