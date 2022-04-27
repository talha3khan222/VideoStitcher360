import cv2

from FeatureExtractor import FeatureExtractor
from Camera import Camera
from MultiCameraStreamer import MultiCameraStreamer


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''image = cv2.imread("Shanghai-Tower-Gensler-San-Francisco-world-Oriental-2015.jpg")
    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fe = FeatureExtractor(name="FAST")
    fe.display_keypoints(gray)

    cv2.imshow('image', gray)
    cv2.waitKey()'''

    # camera_obj = Camera(source=2)
    # camera_obj.stream()

    camera_sources = [3, 2, 4, 0]
    mc = MultiCameraStreamer(camera_sources)
    mc.stream()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
