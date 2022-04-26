import cv2
import datetime
from Descriptors import *


class FeatureExtractor:
    def __init__(self, name="ORB"):
        self._descriptor = None
        class_name = "Descriptor_" + name + "()"
        self._descriptor = eval(class_name)

    def extract_features(self, image):
        key_points, descriptors = self._descriptor.compute_features(image)
        print(len(key_points))
        return key_points, descriptors

    def display_keypoints(self, image):
        key_points, descriptors = self.extract_features(image)

        # draw only key points location,not size and orientation
        img2 = cv2.drawKeypoints(image, key_points, None, color=(0, 255, 0), flags=0)
        cv2.imshow('key points', img2)
