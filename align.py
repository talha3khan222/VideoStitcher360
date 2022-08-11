from __future__ import print_function
import cv2
import numpy as np
from scipy import ndimage as ndi

from Tailor import Tailor


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.75


def registration(P, x_dash, y_dash):
    w1 = np.linalg.inv(P.T @ P) @ P.T @ x_dash
    w2 = np.linalg.inv(P.T @ P) @ P.T @ y_dash
    affine_matrix = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    affine_matrix[0, :] = w1
    affine_matrix[1, :] = w2
    print(affine_matrix)
    return affine_matrix


#A function that refers to the end of the array for those who exceed the range of the reference image
def clip_xy(ref_xy, img_shape):
    #Replace for x coordinate
    ref_x = np.where((0 <= ref_xy[:, 0]) & (ref_xy[:, 0] < img_shape[1]), ref_xy[:, 0], -1)
    #Replace for y coordinate
    ref_y = np.where((0 <= ref_xy[:, 1]) & (ref_xy[:, 1] < img_shape[0]), ref_xy[:, 1], -1)

    #Combine and return
    return np.vstack([ref_x, ref_y]).T


#Affine transformation
def affine(data, affine, draw_area_size):
    # data:Image data to be converted to affine
    # affine:Affine matrix
    #:draw_area_size:It may be the same as or better than the shape of data

    #Inverse matrix of affine matrix
    inv_affine = np.linalg.inv(affine)

    x = np.arange(0, draw_area_size[1], 1)
    y = np.arange(0, draw_area_size[0], 1)
    X, Y = np.meshgrid(x, y)

    XY = np.dstack([X, Y, np.ones_like(X)])
    xy = XY.reshape(-1, 3).T

    #Calculation of reference coordinates
    ref_xy = inv_affine @ xy
    ref_xy = ref_xy.T

    #Coordinates around the reference coordinates
    liner_xy = {}
    liner_xy['downleft'] = ref_xy[:, :2].astype(int)
    liner_xy['upleft'] = liner_xy['downleft'] + [1, 0]
    liner_xy['downright'] = liner_xy['downleft'] + [0, 1]
    liner_xy['upright'] = liner_xy['downleft'] + [1, 1]

    #Weight calculation with linear interpolation
    liner_diff = ref_xy[:, :2] - liner_xy['downleft']

    liner_weight = {}
    liner_weight['downleft'] = (1 - liner_diff[:, 0]) * (1 - liner_diff[:, 1])
    liner_weight['upleft'] = (1 - liner_diff[:, 0]) * liner_diff[:, 1]
    liner_weight['downright'] = liner_diff[:, 0] * (1 - liner_diff[:, 1])
    liner_weight['upright'] = liner_diff[:, 0] * liner_diff[:, 1]

    #Weight and add
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

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    vec_one = np.ones((points1.shape[0], 1))
    P = np.hstack([points1, vec_one])
    x_dash = points2[:, 0]
    y_dash = points2[:, 1]

    A = registration(P, x_dash, y_dash)

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # h[2, :2] = 0

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    img1 = ndi.affine_transform(im1, A)
    # img1 = affine(im2[:, :, 0], A, (480, 480))

    cv2.imshow('Affine', img1)

    combined = cv2.bitwise_or(im1Reg, im2)
    cv2.imshow('Combined', combined)

    return im1Reg, h


if __name__ == '__main__':

    # Read reference image
    refFilename = "images/0.png"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "images/1.png"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    merged = cv2.hconcat([im, imReference])
    cv2.imshow('Merged', merged)

    tailor = Tailor()
    print("Aligning images ...")
    # Registered image will be restored in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)
    # imReg = tailor.align(im, imReference)
    cv2.imshow('Reg', imReg)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    cv2.waitKey()
    cv2.destroyAllWindows()
