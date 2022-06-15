'''import cv2
import numpy as np


img_ = cv2.imread('images/3.png')
#img_ = cv2.imread('original_image_left.jpg')
#img_ = cv2.resize(img_, (0,0), fx=1, fy=1)
img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

img = cv2.imread('images/4.png')
#img = cv2.imread('original_image_right.jpg')
#img = cv2.resize(img, (0,0), fx=1, fy=1)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
# find key points
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))

#FLANN_INDEX_KDTREE = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks = 50)
#match = cv2.FlannBasedMatcher(index_params, search_params)
match = cv2.BFMatcher()
matches = match.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append(m)

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)

img3 = cv2.drawMatches(img_,kp1,img,kp2,good,None,**draw_params)
cv2.imshow("original_image_drawMatches.jpg", img3)
cv2.waitKey()

print(len(good), len(matches))

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    #cv2.imshow("original_image_overlapping.jpg", img2)
else:
    print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))

dst = cv2.warpPerspective(img_, M, (img.shape[1] + img_.shape[1], img.shape[0]))
dst[0:img.shape[0],0:img.shape[1]] = img
cv2.imshow("original_image_stitched.jpg", dst)

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
#cv2.imsave("original_image_stitched_crop.jpg", trim(dst))
cv2.waitKey()

cv2.destroyAllWindows()

'''

import sys
import math
import cv2
from PIL import Image

from Utils.PanoramaBuilder import PanoramaBuilder

from Utils.FeatureMatcher import FeatureMatcher

'''defished = []

defished.append(Image.open("images/0.png"))
defished.append(Image.open("images/6.png"))
defished.append(Image.open("images/4.png"))
defished.append(Image.open("images/2.png"))

pano = PanoramaBuilder()

res = pano.buildPano(defished)

cv2.imshow('Res', res)

cv2.waitKey()

cv2.destroyAllWindows()'''

import numpy as np


def stitch(left, right):
    fm = FeatureMatcher()
    src_pts, dst_pts = fm.match_images(left, right)

    if src_pts is None:
        return

    MIN_MATCH_COUNT = 3
    if len(src_pts) > MIN_MATCH_COUNT:

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homography = M
        h_mask = mask
        print(M, mask)

        h, w = left.shape[0], left.shape[1]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        #img2 = cv2.polylines(right, [np.int32(dst)], True, (0, 255, 255), 2, cv2.LINE_AA)
        #cv2.imshow("original_image_overlapping.jpg", img2)
    else:
        print("Not enough matches found", len(src_pts), MIN_MATCH_COUNT)

    dst = cv2.warpPerspective(left, M, (right.shape[1] + left.shape[1], right.shape[0]))
    cv2.imshow('dst', dst)
    cv2.waitKey()
    dst[0:right.shape[0], 0:right.shape[1]] = right
    cv2.imshow("original_image_stitched.jpg", dst)

    def trim(frame):
        # crop top
        if not np.sum(frame[0]):
            return trim(frame[1:])
        # crop top
        if not np.sum(frame[-1]):
            return trim(frame[:-2])
        # crop top
        if not np.sum(frame[:, 0]):
            return trim(frame[:, 1:])
        # crop top
        if not np.sum(frame[:, -1]):
            return trim(frame[:, :-2])
        return frame

    cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
    cv2.waitKey()


left = cv2.imread("images/4.png")
right = cv2.imread("images/3.png")

# left = left[:, :450]
# right = right[:, :450]

stitch(left, right)

cv2.destroyAllWindows()
