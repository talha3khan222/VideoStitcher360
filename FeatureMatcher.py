from FeatureExtractor import FeatureExtractor
import cv2


class FeatureMatcher:
    def __init__(self):
        self._features_extractor = FeatureExtractor(name="SIFT")
        self._matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher.FLANNBASED)

    def match_images(self, image1, image2):
        kp1, desc1 = self._features_extractor.extract_features(image1)
        kp2, desc2 = self._features_extractor.extract_features(image2)

        if desc1 is not None and desc2 is not None and len(desc1) >= 2 and len(desc2) >= 2:
            raw_match = self._matcher.knnMatch(desc2, desc1, k=2)

        matches = []
        # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
        ratio = 0.75
        for m in raw_match:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            # (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC)
            pass
