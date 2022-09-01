import cv2
import numpy as np
import math

from Utils.generals import apply_affine_transformation


class TransformationAdjuster:
    def __init__(self, img1, img2):

        self._window_name = "Adjust Transformation Parameters"
        cv2.namedWindow(self._window_name)

        self.left_image = img1
        self.right_image = img2

        self._width = img1.shape[1] + img2.shape[1]
        self._height = img1.shape[0]

        self._size = (self._width, self._height)

        self._transformation_matrix = None

        self._current_frame = None

        cv2.createTrackbar('Scale_x', self._window_name, 0, 100, self._nothing)
        cv2.createTrackbar('Scale_y', self._window_name, 0, 100, self._nothing)
        cv2.createTrackbar('Rotation_Angle', self._window_name, 0, 30, self._nothing)
        cv2.createTrackbar('Shear_x', self._window_name, 0, 100, self._nothing)
        cv2.createTrackbar('Shear_y', self._window_name, 0, 100, self._nothing)

        cv2.setTrackbarPos('Scale_x', self._window_name, 50)
        cv2.setTrackbarPos('Scale_y', self._window_name, 50)
        cv2.setTrackbarPos('Rotation_Angle', self._window_name, 15)
        cv2.setTrackbarPos('Shear_x', self._window_name, 50)
        cv2.setTrackbarPos('Shear_y', self._window_name, 50)

    def start(self):
        first_frame = True
        while True:
            frame = self.right_image

            if cv2.waitKey(1) & 0xff == 27:
                break

            # dst = cv2.warpPerspective(frame, self._transformation_matrix, self._size)
            dst = apply_affine_transformation(frame, self._transformation_matrix)

            cv2.imshow(self._window_name, dst)

        cv2.destroyAllWindows()

    def apply_transformations(self, frame, R, T, S):
        dst = apply_affine_transformation(frame, R)

        return dst

    def _nothing(self, val):
        sx = cv2.getTrackbarPos('Scale_x', self._window_name) - 50
        sy = cv2.getTrackbarPos('Scale_y', self._window_name) - 50
        rotation_angle = cv2.getTrackbarPos('Rotation_Angle', self._window_name) - 15
        shx = cv2.getTrackbarPos('Shear_x', self._window_name) - 50
        shy = cv2.getTrackbarPos('Shear_y', self._window_name) - 50

        S = np.array([[sx, 0, 0],
                      [0, sy, 0],
                      [0, 0, 1]])

        R = np.array([[math.cos(rotation_angle), -math.sin(rotation_angle), 0],
                      [math.sin(rotation_angle), math.cos(rotation_angle), 0],
                      [0, 0, 1]])

        T = np.array([[0, shx, 0],
                      [shy, 0, 0],
                      [0, 0, 1]])

        TS = np.matmul(T, S)
        A = np.matmul(R, TS)

        self._transformation_matrix = R


left = cv2.imread("images/2.png")
right = cv2.imread("images/q1.png")

width = left.shape[1] + right.shape[1]
height = left.shape[0]

right = cv2.resize(right, (left.shape[1], left.shape[0]))

tp = TransformationAdjuster(left, right)
tp.start()

