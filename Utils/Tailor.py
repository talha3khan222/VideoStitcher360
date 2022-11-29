import cv2
import numpy as np
from scipy import ndimage as ndi

from Utils.FeatureMatcher import FeatureMatcher
from Utils.generals import registration, get_angle, apply_homography ,apply_affine_transformation


class Tailor:
    def __init__(self):
        self._feature_matcher = FeatureMatcher()
        self._transformation_matrix = None
        self._image_points = None

    def align(self, img1, img2):
        points1, points2, imMatches = self._feature_matcher.get_orb_matching_points(img1, img2)

        t_matrix = self.compute_transformations(points1, points2)

        transformed_images = self.apply_transformations(img2, t_matrix, (480, 480))

        return transformed_images

    def compute_transformations(self, points1, points2):
        vec_one = np.ones((points1.shape[0], 1))
        P = np.hstack([points1, vec_one])
        x_dash = points2[:, 0]
        y_dash = points2[:, 1]

        A = registration(P, x_dash, y_dash)
        print("Estimated Affine Transformation: \n", A)

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        print("Estimated Homography: \n", h)

        transformation_matrix = {"homography": h, "affine": A}

        return transformation_matrix

    def apply_transformations(self, img, transformation_matrix, shape):
        A_inv = np.linalg.inv(transformation_matrix["affine"])
        transformed_images = {"homography": apply_homography(img, transformation_matrix["homography"], shape),
                              "affine": apply_affine_transformation(img, A_inv)}

        return transformed_images

    def calculate_required_parameters(self, img1, img2):
        points1, points2 = self._feature_matcher.get_orb_matching_points(img1, img2)

        self._transformation_matrix = self.compute_transformations(points1, points2)

        pt1 = points1[0]
        pt2 = points2[0]

        new_pt1 = [pt1[0] + img1.shape[1] // 2, pt1[1] + img1.shape[1]]

        x = np.array([[pt2[0]], [pt2[1]], [1]])
        x_dash = np.matmul(np.linalg.inv(self._transformation_matrix["affine"]), x)
        pt2 = x_dash[:2, 0]

        image_starting = (int(new_pt1[0] - pt2[0]), 0)
        image_ending = (int(image_starting[0] + img2.shape[1]), int(image_starting[1] + img2.shape[0]))

        self._image_points = {"starting": image_starting, "ending": image_ending}

    def stitch(self, img1, img2):
        if self._transformation_matrix is None:
            self.calculate_required_parameters(img1, img2)

        img = apply_affine_transformation(img2, self._transformation_matrix["affine"])
        cv2.imshow('img', img)
        cv2.waitKey()

        final = np.zeros((480, 960, 3), dtype=np.uint8)
        final[self._image_points["starting"][1]: self._image_points["ending"][1],
              self._image_points["starting"][0]: self._image_points["ending"][0]] = img

        final[:, :480] = img1

        return final

    def display_matching_points(self, image1, image2):
        p1, p2 = self._feature_matcher.get_orb_matching_points(image1, image2)
        return None


t = Tailor()
left = cv2.imread("../images/1.png")
right = cv2.imread("../images/2.png")

left = cv2.resize(left, (right.shape[1], right.shape[0]))

st = t.stitch(left, right)

cv2.imshow('stitched', st)
cv2.waitKey()

cv2.destroyAllWindows()
