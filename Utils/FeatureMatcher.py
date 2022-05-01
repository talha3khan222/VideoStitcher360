from Utils.FeatureExtractor import FeatureExtractor
import cv2
import numpy as np


class FeatureMatcher:
    def __init__(self):
        self._features_extractor = FeatureExtractor(name="ORB", num_features=1000)
        # self._matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher.FLANNBASED)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._matching_threshold = 0.75
        self._minimum_matching_points = 4

    def match_images(self, image1, image2):
        kp1, desc1 = self._features_extractor.extract_features(image1)
        kp2, desc2 = self._features_extractor.extract_features(image2)

        if desc1 is not None and desc2 is not None and len(desc1) >= 2 and len(desc2) >= 2:
            raw_matches = self._matcher.knnMatch(desc1, desc2, k=2)
            # raw_matches = self._matcher.radiusMatch(desc1, desc2, 2)

            good_matches = []
            # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            for match_1, match_2 in raw_matches:
                if match_1.distance < self._matching_threshold * match_2.distance:
                    good_matches.append(match_1)

            if len(good_matches) < self._minimum_matching_points:
                return None, None

            # filter good matching key points
            good_kps_1 = []
            good_kps_2 = []

            for match in good_matches:
                good_kps_1.append(kp1[match.queryIdx].pt)  # matching keypoint in image 1
                good_kps_2.append(kp2[match.trainIdx].pt)  # matching keypoint in image 2

            return np.array(good_kps_1), np.array(good_kps_2)

