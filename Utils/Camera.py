import cv2


class Camera:
    def __init__(self, source=0):
        self._name = str(source)
        self._source = source
        self._capturing_handle = cv2.VideoCapture(self._source)
        self._keep_streaming = True

    def get_name(self):
        return self._name

    def release_handle(self):
        self._capturing_handle.release()

    def grab_frame(self, ret_value):
        ret, frame = self._capturing_handle.read()

        ret_value = frame if ret else None

        return ret_value

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
