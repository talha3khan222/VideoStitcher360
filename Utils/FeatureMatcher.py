from Utils.FeatureExtractor import FeatureExtractor
import cv2
import numpy as np


class FeatureMatcher:
    def __init__(self):
        self._features_extractor = FeatureExtractor(name="SIFT", num_features=1000)
        # self._matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher.FLANNBASED)
        self._matcher = cv2.BFMatcher()

        self._matching_threshold = 0.5
        self._minimum_matching_points = 3

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
                print("Not Enough Matching Points", len(good_matches))
                return None, None

            # filter good matching key points
            good_kps_1 = []
            good_kps_2 = []
            
            good_kps_1 = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
            good_kps_2 = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)

            '''for match in good_matches:
                good_kps_1.append(kp1[match.queryIdx].pt)  # matching keypoint in image 1
                good_kps_2.append(kp2[match.trainIdx].pt)  # matching keypoint in image 2'''
            

            return good_kps_1, good_kps_2

