import cv2
import numpy as np

from Utils.generals import combine_images, compute_affine_transformation


def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = (0, 255, 255) if data['name'] == "Image" else (0, 0, 255)
        cv2.circle(data['im'], (x, y), 2, color, 3)
        cv2.imshow(data['name'], data['im'])
        # if len(data['points']) < 4:
        data['points'].append([x, y])


def get_four_points(im, imRef):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    data['name'] = "Image"

    dataRef = {}
    dataRef['im'] = imRef.copy()
    dataRef['points'] = []
    dataRef['name'] = "ImageRef"

    # Set the callback function for any mouse event
    cv2.imshow("Image", im)
    cv2.imshow("ImageRef", imRef)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.setMouseCallback("ImageRef", mouse_handler, dataRef)
    cv2.waitKey(0)

    # Convert array to np.array
    points = np.vstack(data['points']).astype(int)
    pointsRef = np.vstack(dataRef['points']).astype(int)

    return points, pointsRef


from Utils.FeatureMatcher import FeatureMatcher


left = cv2.imread("images/0.png")
right = cv2.imread("images/1.png")

# right = cv2.resize(right, (max([left.shape[1], right.shape[1]]), left.shape[0]))

flipped_left = cv2.flip(left, 1)
left = cv2.flip(right, 1)

right = flipped_left


src_pts, ref_pts = get_four_points(left, right)

tform, status = cv2.findHomography(src_pts, ref_pts)

A = compute_affine_transformation(src_pts, ref_pts)

print("Homography : ", tform)
print("Affine Transformation : ", A)

cv2.imshow("im_H", combine_images(right, left, np.linalg.inv(tform)))
cv2.imshow("im_A", combine_images(right, left, np.linalg.inv(A)))

cv2.waitKey()

cv2.destroyAllWindows()
