import cv2
from Utils.DeFisheye import DeFishEye
import numpy as np


class Camera:
    def __init__(self, source=0, doUnwarp=False, doCrop=False, parameters={}):
        self._name = str(source)
        self._source = source
        self._capturing_handle = cv2.VideoCapture(self._source)

        #self._capturing_handle.set(3, 1920)
        #self._capturing_handle.set(4, 1080)
        
        self._keep_streaming = True
        
        self._defish = DeFishEye(160, 160)
        self._first_frame = True
        self._doUnwarp = doUnwarp
        self._doCrop = doCrop

        self._parameters = parameters

        white_indices = np.where(self._parameters['mask'][:, :, 0] == 255)

        self._startX = min(white_indices[0])
        self._startY = min(white_indices[1])

        self._endX = max(white_indices[0])
        self._endY = max(white_indices[1])

        self.frame_size = None

    def get_name(self):
        return self._name

    def release_handle(self):
        self._capturing_handle.release()

    def grab_frame(self, ret_value):
        ret, frame = self._capturing_handle.read()

        if not ret:
            return None

        frame = cv2.bitwise_and(frame, self._parameters['mask'])
        frame = frame[self._startX:self._endX, self._startY: self._endY]

        frame = cv2.resize(frame, (480, 480))
        
        if self._doUnwarp:
            cv2.imwrite("images/" + self._name + "_fisheye.png", frame)
            if self._first_frame:
                Ws = frame.shape[1]
                Hs = frame.shape[0]
                Wd = round(frame.shape[1] * (4.0 / 3.0))
                Hd = frame.shape[0]

                self._defish.buildmap(Ws, Hs, Ws, Hd)
                self._first_frame = False

            frame = self._defish.unwarp(frame, self._doCrop)

        return frame

    def stream(self):
        while self._keep_streaming:
            ret, frame = self._capturing_handle.read()
            if ret:
                cv2.imshow('frame', frame)
            else:
                print("Nothing")

            key = cv2.waitKey(1)
            if key == ord('q'):
                self._keep_streaming = False

