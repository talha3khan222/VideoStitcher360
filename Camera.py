import cv2


class Camera:
    def __init__(self, source=0):
        self._source = source
        self._capturing_handle = cv2.VideoCapture(self._source)
        self._keep_streaming = True

    def grab_frame(self):
        ret, frame = self._capturing_handle.read()

        if ret:
            return frame

        return None

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
