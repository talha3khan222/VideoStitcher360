import cv2
import datetime
from Descriptors import *


class Descriptor_SIFT:
    def __init__(self):
        self.detector = cv2.SIFT_create()

    def compute_features(self, image):
        # Find keypoints and descriptors directly
        kp, des = self.detector.detectAndCompute(image, None)

        return kp


class FeatureExtractor:
    def __init__(self, name="ORB"):
        self._descriptor = None
        class_name = "Descriptor_" + name + "()"
        self._descriptor = eval(class_name)

    def compute_keypoints(self, image):
        key_points = self._descriptor.compute_features(image)

        print(len(key_points))

        return key_points

    def display_keypoints(self, image):
        start = datetime.datetime.now()
        key_points = self.compute_keypoints(image)
        end = datetime.datetime.now()

        print("Time Taken: ", (end-start).total_seconds())

        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(image, key_points, None, color=(0, 255, 0), flags=0)
        cv2.imshow('key points', img2)
