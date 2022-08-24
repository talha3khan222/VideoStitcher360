from __future__ import print_function
import cv2
import numpy as np
from scipy import ndimage as ndi

from Utils.Tailor import Tailor
from Utils.generals import registration, get_angle


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


# A function that refers to the end of the array for those who exceed the range of the reference image
def clip_xy(ref_xy, img_shape):
    # Replace for x coordinate
    ref_x = np.where((0 <= ref_xy[:, 0]) & (ref_xy[:, 0] < img_shape[1]), ref_xy[:, 0], -1)
    # Replace for y coordinate
    ref_y = np.where((0 <= ref_xy[:, 1]) & (ref_xy[:, 1] < img_shape[0]), ref_xy[:, 1], -1)

    # Combine and return
    return np.vstack([ref_x, ref_y]).T


# Affine transformation
def affine(data, affine, draw_area_size):
    # data:Image data to be converted to affine
    # affine:Affine matrix
    # :draw_area_size:It may be the same as or better than the shape of data

    # Inverse matrix of affine matrix
    inv_affine = np.linalg.inv(affine)

    x = np.arange(0, draw_area_size[1], 1)
    y = np.arange(0, draw_area_size[0], 1)
    X, Y = np.meshgrid(x, y)

    XY = np.dstack([X, Y, np.ones_like(X)])
    xy = XY.reshape(-1, 3).T

    # Calculation of reference coordinates
    ref_xy = inv_affine @ xy
    ref_xy = ref_xy.T

    # Coordinates around the reference coordinates
    liner_xy = {}
    liner_xy['downleft'] = ref_xy[:, :2].astype(int)
    liner_xy['upleft'] = liner_xy['downleft'] + [1, 0]
    liner_xy['downright'] = liner_xy['downleft'] + [0, 1]
    liner_xy['upright'] = liner_xy['downleft'] + [1, 1]

    # Weight calculation with linear interpolation
    liner_diff = ref_xy[:, :2] - liner_xy['downleft']

    liner_weight = {}
    liner_weight['downleft'] = (1 - liner_diff[:, 0]) * (1 - liner_diff[:, 1])
    liner_weight['upleft'] = (1 - liner_diff[:, 0]) * liner_diff[:, 1]
    liner_weight['downright'] = liner_diff[:, 0] * (1 - liner_diff[:, 1])
    liner_weight['upright'] = liner_diff[:, 0] * liner_diff[:, 1]

    # Weight and add
    liner_with_weight = {}
    for direction in liner_weight.keys():
        l_xy = liner_xy[direction]
        l_xy = clip_xy(l_xy, data.shape)
        l_xy = np.dstack([l_xy[:, 0].reshape(draw_area_size), l_xy[:, 1].reshape(draw_area_size)])
        l_weight = liner_weight[direction].reshape(draw_area_size)
        liner_with_weight[direction] = data[l_xy[:, :, 1], l_xy[:, :, 0]] * l_weight

    data_linear = sum(liner_with_weight.values())
    return data_linear


def alignImages(im1, im2):

    cim1 = im1.copy()
    cim2 = im2.copy()

    im1 = im1[:, (im1.shape[1] // 2):]
    im2 = im2[:, :(im2.shape[1] // 2)]

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    feature_extractor = cv2.ORB_create(MAX_FEATURES)
    # feature_extractor = cv2.SIFT_create(MAX_FEATURES)

    keypoints1, descriptors1 = feature_extractor.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = feature_extractor.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    # FLANN parameters
    '''FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)'''

    matches = matcher.match(descriptors1, descriptors2, None)
    # matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Sort matches by score
    list(matches).sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    cv2.imshow('Matches', imMatches)
    # cv2.waitKey()

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    points1 = []
    points2 = []

    for i, match in enumerate(matches):

        p2 = keypoints2[match.trainIdx].pt

        left_point = list(keypoints1[match.queryIdx].pt)
        right_point = [p2[0], p2[1]]

        angle = get_angle(left_point, right_point)

        print(left_point, right_point, angle)

        if abs(angle) < 60:
            print(angle)
            points1.append(keypoints1[match.queryIdx].pt)
            points2.append(keypoints2[match.trainIdx].pt)
        # points1[i, :] = keypoints1[match.queryIdx].pt
        # points2[i, :] = keypoints2[match.trainIdx].pt

    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)

    ########################################################################

    vec_one = np.ones((points1.shape[0], 1))
    P = np.hstack([points1, vec_one])
    x_dash = points2[:, 0]
    y_dash = points2[:, 1]

    A = registration(P, x_dash, y_dash)

    A_inv = np.linalg.inv(A)

    img2 = cv2.merge([ndi.affine_transform(cim1[:, :, 0], A_inv),
                      ndi.affine_transform(cim1[:, :, 1], A_inv),
                      ndi.affine_transform(cim1[:, :, 2], A_inv)])
    # img1 = affine(im2[:, :, 0], A, (480, 480))

    cv2.imshow('Inverse Affine', img2)

    ########################################################################

    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    print("Estimated Homography: \n", H)

    # Use homography
    height, width, channels = cim1.shape
    im1Reg = cv2.warpPerspective(cim1, H, (width, height))
    cv2.imshow('Homography', im1Reg)

    ########################################################################

    pt1 = points1[0]
    pt2 = points2[0]

    new_pt1 = [pt1[0] + cim1.shape[1] // 2, pt1[1] + cim1.shape[1]]

    x = np.array([[pt2[0]], [pt2[1]], [1]])
    x_dash = np.matmul(A_inv, x)
    pt2 = x_dash[:2, 0]

    image_starting = (int(new_pt1[0] - pt2[0]), 0)
    image_ending = (int(image_starting[0] + img2.shape[1]), int(image_starting[1] + img2.shape[0]))

    img1 = cv2.merge([ndi.affine_transform(cim1[:, :, 0], A),
                      ndi.affine_transform(cim1[:, :, 1], A),
                      ndi.affine_transform(cim1[:, :, 2], A)])

    cv2.imshow("Affine", img1)

    '''final = np.zeros((480, 960, 3), dtype=np.uint8)
    final[image_starting[1]: image_ending[1], image_starting[0]:image_ending[0]] = img1
    final[:, :480] = cim1'''

    dst = cv2.warpPerspective(cim2, H, (cim1.shape[1] + cim2.shape[1], cim2.shape[0]))
    dst[0:cim2.shape[0], 0:cim2.shape[1]] = cim1
    cv2.imshow('dst', dst)

    # final = cv2.rectangle(final, image_starting, image_ending, (0, 255, 255), 2)

    # cv2.imshow('Final', final)

    return im1Reg, H


if __name__ == '__main__':

    # Read reference image
    refFilename = "images/0.png"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "images/3.png"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    merged = cv2.hconcat([im, imReference])
    cv2.imshow('Merged', merged)

    tailor = Tailor()
    print("Aligning images ...")
    # Registered image will be restored in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # stitched_image = tailor.stitch(im, imReference)
    # cv2.imshow('Stitched', stitched_image)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    # cv2.imwrite(outFilename, imReg)

    cv2.waitKey()
    cv2.destroyAllWindows()
