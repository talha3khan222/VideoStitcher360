import cv2
import numpy as np


class ParametersAdjuster:
    def __init__(self, streaming_source=0):
        self._streaming_source = streaming_source

        self._streaming_handle = cv2.VideoCapture(streaming_source)

        self._center = None
        self._radius = 0

        self._window_name = str(streaming_source) + " Adjust Parameters"
        cv2.namedWindow(self._window_name)

        width = round(self._streaming_handle.get(3))
        height = round(self._streaming_handle.get(4))

        self._frame_size = (width, height)

        cv2.createTrackbar('center_x', self._window_name, 0, width, self._nothing)
        cv2.createTrackbar('center_y', self._window_name, 0, height, self._nothing)
        cv2.createTrackbar('radius', self._window_name, 0, width // 2, self._nothing)

        cv2.setTrackbarPos('center_x', self._window_name, width // 2)
        cv2.setTrackbarPos('center_y', self._window_name, height // 2)
        cv2.setTrackbarPos('radius', self._window_name, 10)

    def start(self):
        first_frame = True
        while True:
            ret, frame = self._streaming_handle.read()

            if not ret or (cv2.waitKey(1) & 0xff == 27):
                break
            if first_frame:
                self._radius = 10
                self._center = (frame.shape[1] // 2, frame.shape[0] // 2)
                first_frame = False

            cx = cv2.getTrackbarPos('center_x', self._window_name)
            cy = cv2.getTrackbarPos('center_y', self._window_name)
            radius = cv2.getTrackbarPos('radius', self._window_name)

            self._radius = radius
            self._center = (cx, cy)

            cv2.circle(frame, self._center, self._radius, (0, 255, 255), 1)

            cv2.imshow(self._window_name, frame)

        self._streaming_handle.release()
        cv2.destroyAllWindows()

    def _nothing(self, val):
        pass

    def get_parameters(self):
        mask = np.zeros((self._frame_size[1], self._frame_size[0], 3), dtype=np.uint8)
        mask = cv2.circle(mask, self._center, self._radius, (255, 255, 255), -1)
        return {'radius': self._radius, 'center': self._center, 'mask': mask}


def adjust_all_camera_parameters(streaming_urls):
    parameters = []
    for source in streaming_urls:
        pa = ParametersAdjuster(source)
        pa.start()
        parameters.append(pa.get_parameters())

    return parameters

