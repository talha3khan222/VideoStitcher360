import cv2
from Utils.DeFisheye import DeFishEye


class Camera:
    def __init__(self, source=0, doUnwarp=False, doCrop=False):
        self._name = str(source)
        self._source = source
        self._capturing_handle = cv2.VideoCapture(self._source)
        
        self._keep_streaming = True
        
        self._defish = DeFishEye(160, 160)
        self._first_frame = True
        self._doUnwarp = doUnwarp
        self._doCrop = doCrop
        
        self._startY, self._endY = 0, 640
        self._startX, self._endX = 0, 480
        if source == 1:
            self._startY = 97
            self._endY = 595
        elif source == 2:
            self._startX = 10
            self._startY = 90
            self._endY = 571
        elif source == 3:
            self._startX = 15
            self._startY = 73
            self._endY = 570
        elif source == 4:
            self._startY = 85
            self._endY = 567

    def get_name(self):
        return self._name

    def release_handle(self):
        self._capturing_handle.release()

    def grab_frame(self, ret_value):
        ret, frame = self._capturing_handle.read()

        if not ret:
            return None

        frame = frame[self._startX:self._endX, self._startY: self._endY]
        
        if self._doUnwarp:
            cv2.imwrite("images/" + self._name + "_fisheye.png", frame)
            if self._first_frame:
                Ws = frame.shape[1]
                Hs = frame.shape[0]
                Wd = round(frame.shape[1] * (4.0 / 3.0))
                Hd = frame.shape[0]

                self._defish.buildmap(Ws, Hs, Ws, Hd)
                self._first_frame = False
            else:
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
