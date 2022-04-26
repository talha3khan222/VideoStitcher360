import cv2

from FeatureExtractor import FeatureExtractor


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image = cv2.imread("Shanghai-Tower-Gensler-San-Francisco-world-Oriental-2015.jpg")
    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fe = FeatureExtractor(name="FAST")
    fe.display_keypoints(gray)

    cv2.imshow('image', gray)
    cv2.waitKey()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
