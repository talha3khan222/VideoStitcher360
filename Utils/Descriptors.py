import cv2


class Descriptor_ORB:
    def __init__(self, number_of_features):
        self.detector = cv2.ORB_create(number_of_features)

    def compute_features(self, image):
        kp = self.detector.detect(image, None)

        # compute the descriptors with ORB
        kp, des = self.detector.compute(image, kp)

        return kp, des


class Descriptor_FAST:
    def __init__(self, number_of_features):
        # self.detector = cv2.xfeatures2d.StarDetector_create(number_of_features)
        self.detector = cv2.FastFeatureDetector(number_of_features)
        # self.extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        #self.extractor = cv2.BOWImgDescriptorExtractor(self.extractor)

    def compute_features(self, image):
        # find the keypoints with STAR
        kp = self.detector.detect(image, None)
        kp, des = self.detector.detectAndCompute(image, None)

        # compute the descriptors with BRIEF
        #kp, des = self.extractor.compute(image, kp)

        return kp, des


class Descriptor_SURF:
    def __init__(self, number_of_features):
        self.detector = cv2.xfeatures2d.SURF_create(number_of_features)

    def compute_features(self, image):
        # Find keypoints and descriptors directly
        kp, des = self.detector.detectAndCompute(image, None)

        return kp, des


class Descriptor_SIFT:
    def __init__(self, number_of_features):
        self.detector = cv2.SIFT_create(number_of_features)

    def compute_features(self, image):
        # Find keypoints and descriptors directly
        kp, des = self.detector.detectAndCompute(image, None)

        return kp, des
