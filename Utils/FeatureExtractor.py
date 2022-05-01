import cv2
import datetime
from Utils.Descriptors import *


class FeatureExtractor:
    def __init__(self, name="ORB", num_features=400):
        self._descriptor = None
        class_name = "Descriptor_" + name + "(" + str(num_features) + ")"
        self._descriptor = eval(class_name)

    def extract_features(self, image):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = self._descriptor.compute_features(grayscale)
        return key_points, descriptors

    def display_keypoints(self, image):
        key_points, descriptors = self.extract_features(image)

        # draw only key points location,not size and orientation
        img2 = cv2.drawKeypoints(image, key_points, None, color=(0, 255, 0), flags=0)
        cv2.imshow('key points', img2)
