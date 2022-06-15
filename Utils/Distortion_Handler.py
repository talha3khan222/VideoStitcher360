from PIL import Image
from PIL.ExifTags import TAGS
import lensfunpy  # Lensfun
import cv2  # OpenCV library
import os
from multiprocessing import Pool
import timeit  # Add a timer
import piexif


class Distortion_Handler:
    def __init__(self):
        cam_maker = "GoPro"
        cam_model = "HD2"

        db = lensfunpy.Database()
        self._cam = db.find_cameras(cam_maker, cam_model)[0]
        self._lens = db.find_lenses(self._cam)[0]

        # TODO: set camera parameters from exif data and lensfun
        self._focalLength = self._lens.min_focal  # 2.5
        self._aperture = 3.8
        self._distance = 0

    def correct_image(self, image):
        height, width = image.shape[0], image.shape[1]

        mod = lensfunpy.Modifier(self._lens, self._cam.crop_factor - 1.2, int(width * 1.0), int(height * 1.0))
        mod.initialize(self._focalLength, self._aperture, self._distance, 0.0)

        undistCoords = mod.apply_geometry_distortion()
        # imUndistorted = cv2.remap(im, undistCoords, None, cv2.INTER_LANCZOS4)
        undistorted_image = cv2.remap(image, undistCoords, None, cv2.INTER_NEAREST)

        return undistorted_image
