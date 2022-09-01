import cv2
import numpy as np
from math import atan2, degrees, radians
from scipy import ndimage as ndi


def find_cameras_indexes(count_looking_for=4):
    max_index = count_looking_for * 6
    cam_index = 0
    found_camera_indexes = []
    while cam_index < max_index and len(found_camera_indexes) < count_looking_for:
        cap = cv2.VideoCapture(cam_index)
        if cap.read()[0]:
            found_camera_indexes.append(cam_index)
            cap.release()
        cam_index += 1

    return found_camera_indexes


def registration(P, x_dash, y_dash):
    w1 = np.linalg.inv(P.T @ P) @ P.T @ x_dash
    w2 = np.linalg.inv(P.T @ P) @ P.T @ y_dash
    affine_matrix = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    affine_matrix[0, :] = w1
    affine_matrix[1, :] = w2
    return affine_matrix


def compute_affine_transformation(points1, points2):
    vec_one = np.ones((points1.shape[0], 1))
    P = np.hstack([points1, vec_one])
    x_dash = points2[:, 0]
    y_dash = points2[:, 1]

    A = registration(P, x_dash, y_dash)
    return A


def get_angle(point_1, point_2):  # These can also be four parameters instead of two arrays
    angle = atan2(point_2[1] - point_1[1], point_2[0] - point_1[0])

    # Optional
    angle = degrees(angle)

    # OR
    # angle = radians(angle)

    return angle


def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop top
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # crop top
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def apply_homography(img, matrix, shape):
    return cv2.warpPerspective(img, matrix, shape)


def combine_images(left, right, matrix):
    dst = cv2.warpPerspective(left, matrix, (right.shape[1] + left.shape[1], right.shape[0]))
    # cv2.imshow('dst', dst)
    # cv2.waitKey()
    dst[0:right.shape[0], 0:right.shape[1]] = right

    return trim(dst)


def apply_affine_transformation(img, matrix):
    transformed_image = cv2.merge([ndi.affine_transform(img[:, :, 0], matrix),
                                   ndi.affine_transform(img[:, :, 1], matrix),
                                   ndi.affine_transform(img[:, :, 2], matrix)])

    return transformed_image
