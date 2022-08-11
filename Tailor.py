import cv2
import numpy as np
from Utils.FeatureMatcher import FeatureMatcher


def registration(P, x_dash, y_dash):
    w1 = np.linalg.inv(P.T @ P) @ P.T @ x_dash
    w2 = np.linalg.inv(P.T @ P) @ P.T @ y_dash
    affine_matrix = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    affine_matrix[0, :] = w1
    affine_matrix[1, :] = w2

    return affine_matrix


class Tailor:
    def __init__(self):
        self._feature_matcher = FeatureMatcher()

    def align(self, img1, img2):
        points1, points2 = self._feature_matcher.get_matching_points(img1, img2)

        t_matrix = self.compute_transformations(points1, points2)

        transformed_images = self.apply_transformations(img2, t_matrix, (480, 480))

        return transformed_images["homography"]

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

    def apply_homography(self, img, h, shape):
        imReg = cv2.warpPerspective(img, h, shape)

        '''h, w, _ = img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, h)

        cv2.imshow('dst', dst)'''

        return imReg

    def apply_affine_transformation(self, img, matrix, shape):
        return None

    def apply_transformations(self, img, transformation_matrix, shape):
        transformed_images = {"homography": self.apply_homography(img, transformation_matrix["homography"], shape),
                              "affine": self.apply_affine_transformation(img, transformation_matrix["affine"], shape)}

        return transformed_images

