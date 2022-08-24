from __future__ import print_function
import cv2
import numpy as np
from scipy import ndimage as ndi

from Utils.Tailor import Tailor
from Utils.generals import registration, get_angle, apply_affine_transformation, apply_homography


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


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

    matches = matcher.match(descriptors1, descriptors2, None)
    # matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Sort matches by score
    # list(matches).sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    # numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    # matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imshow('Matches', imMatches)
    # cv2.waitKey()

    # Extract location of good matches
    # points1 = np.zeros((len(matches), 2), dtype=np.float32)
    # points2 = np.zeros((len(matches), 2), dtype=np.float32)

    points1 = []
    points2 = []

    for i, match in enumerate(matches):

        p2 = keypoints2[match.trainIdx].pt

        left_point = list(keypoints1[match.queryIdx].pt)
        right_point = [p2[0]+240, p2[1]]

        angle = get_angle(left_point, right_point)

        # print(left_point, right_point, angle)

        if abs(angle) < 60:
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
    print("Estimated Affine Matrix", A)

    ########################################################################

    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    print("Estimated Homography: \n", H)
    H[0, 2] += 240

    # Use homography
    height, width, channels = cim2.shape
    im1Reg = cv2.warpPerspective(cim2, H, (width * 2, height))
    im1Reg[0:cim2.shape[0], 0:cim2.shape[1]] = cim1
    cv2.imshow('Homography', im1Reg)

    ########################################################################

    img1 = apply_affine_transformation(cim1, A)

    cv2.imshow("Affine", img1)

    return im1Reg, H


if __name__ == '__main__':

    # Read reference image
    refFilename = "images/1.png"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "images/0.png"
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
