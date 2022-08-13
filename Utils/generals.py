import cv2
import numpy as np
from math import atan2, degrees, radians


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
    print(affine_matrix)
    return affine_matrix


def get_angle(point_1, point_2):  # These can also be four parameters instead of two arrays
    angle = atan2(point_2[1] - point_1[1], point_2[0] - point_1[0])

    # Optional
    angle = degrees(angle)

    # OR
    # angle = radians(angle)

    return angle

