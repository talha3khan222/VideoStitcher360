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


left = cv2.imread("images/2.png")
right = cv2.imread("images/q1.png")

width = left.shape[1] + right.shape[1]
height = left.shape[0]

right = cv2.resize(right, (left.shape[1], left.shape[0]))

# src_pts, ref_pts = get_four_points(left, right)

# tform, status = cv2.findHomography(src_pts, ref_pts)
tform = np.array([[ 3.72578228e-01, -1.64889511e-01, -2.90926637e+01],
                  [-9.71337367e-02,  4.53838987e-01,  4.89881873e+01],
                  [-1.00823278e-03, -3.91080682e-04,  1.00000000e+00]])

# A = compute_affine_transformation(src_pts, ref_pts)

A = np.array([[ 2.07098001e+00, -7.74358795e-01, -7.85347120e+02],
              [ 3.96727382e-01,  9.31403840e-01, -1.62089883e+02],
              [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

print(tform)
print(A)

cv2.imshow("im_H", combine_images(right, left, np.linalg.inv(tform)))
cv2.imshow("im_A", combine_images(right, left, np.linalg.inv(A)))

cv2.waitKey()

cv2.destroyAllWindows()
