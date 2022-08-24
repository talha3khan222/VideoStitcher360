from Utils.FeatureExtractor import FeatureExtractor
import cv2
import numpy as np

from Utils.generals import get_angle


class FeatureMatcher:
    def __init__(self):
        self._sift_features_extractor = FeatureExtractor(name="SIFT", num_features=1000)
        self._orb_features_extractor = FeatureExtractor(name="ORB", num_features=1000)
        self._knn_matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self._matching_threshold = 0.5
        self._minimum_matching_points = 3

    def match_images(self, image1, image2):
        kp1, desc1 = self._sift_features_extractor.extract_features(image1)
        kp2, desc2 = self._sift_features_extractor.extract_features(image2)

        if desc1 is not None and desc2 is not None and len(desc1) >= 2 and len(desc2) >= 2:
            raw_matches = self._knn_matcher.knnMatch(desc1, desc2, k=2)
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
            
            good_kps_1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            good_kps_2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            return good_kps_1, good_kps_2

    def get_matching_points(self, im1, im2):
        cim1 = im1.copy()
        cim2 = im2.copy()

        im1 = im1[:, (im1.shape[1] // 2):]
        im2 = im2[:, :(im2.shape[1] // 2)]

        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        keypoints1, descriptors1 = self._orb_features_extractor.extract_features(im1)
        keypoints2, descriptors2 = self._orb_features_extractor.extract_features(im2)

        matches = self._bf_matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        list(matches).sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self._matching_threshold)
        matches = matches[:numGoodMatches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        points1 = []
        points2 = []

        for i, match in enumerate(matches):
            p2 = keypoints2[match.trainIdx].pt
            angle = get_angle(list(keypoints1[match.queryIdx].pt), [p2[0] + 480, p2[1] + 480])

            if abs(angle) < 60:
                print(angle)
                points1.append(keypoints1[match.queryIdx].pt)
                points2.append(keypoints2[match.trainIdx].pt)

        points1 = np.array(points1, dtype=np.float32)
        points2 = np.array(points2, dtype=np.float32)

        return points1, points2
