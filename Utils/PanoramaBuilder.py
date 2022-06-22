import cv2
import numpy as np
from PIL import Image



class PanoramaBuilder:
    def __init__(self, quality=500, distance_thresh=90, matching_thresh=0.2):
        self._quality = quality
        self._minDist = distance_thresh
        self._minMatch = matching_thresh

        self._offets = []
        self._homography_matricx = []

    def findHomography(self, img, template):
        # cribbed from SimpleCV, the homography sucks for this
        # just use the median of the x offset of the keypoint correspondences
        # to determine how to align the image

        d = cv2.ORB_create(int(self._quality))
        skp, sd = d.detectAndCompute(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY), None)
        tkp, td = d.detectAndCompute(cv2.cvtColor(np.array(template), cv2.COLOR_BGR2GRAY), None)

        if skp is None or tkp is None:
            Warning.warn("I didn't get any keypoints. Image might be too uniform or blurry.")
            return None

        magic_ratio = sd.shape[0] / td.shape[0] if sd.shape[0] > td.shape[0] else 1

        m = cv2.BFMatcher(cv2.NORM_HAMMING)
        rawMatches = m.knnMatch(sd, td, 2)

        idx = []
        dist = []
        result = []

        min_range = min([len(skp), len(tkp)])

        for count, (match_1, match_2) in enumerate(rawMatches):
            if count < min_range and match_1.trainIdx < min_range:
                dist.append([match_1.distance, match_2.distance])
                idx.append(match_1.trainIdx)
                result.append(False)
                if match_1.distance < magic_ratio * match_2.distance:
                    result[-1] = True

        dist = np.array(dist)

        p = dist[:, 0]
        result = p * magic_ratio < self._minDist
        pr = sum(result) / float(dist.shape[0])

        if pr > self._minMatch and sum(result) > 4:  # if more than minMatch % matches we go ahead and get the data
            # FIXME this code computes the "correct" homography
            lhs = []
            rhs = []
            for i in range(0, len(idx)):
                if result[i]:
                    try:
                        lhs.append((tkp[i].pt[1], tkp[i].pt[0]))  # FIXME note index order
                        rhs.append((skp[idx[i]].pt[0], skp[idx[i]].pt[1]))  # FIXME note index order
                    except Exception as e:
                        print(i, len(skp), len(tkp), len(idx), idx[i])

            rhs_pt = np.array(rhs)
            lhs_pt = np.array(lhs)
            xm = np.median(rhs_pt[:, 1] - lhs_pt[:, 1])
            ym = np.median(rhs_pt[:, 0] - lhs_pt[:, 0])
            homography, mask = cv2.findHomography(lhs_pt, rhs_pt, cv2.RANSAC, ransacReprojThreshold=1.1)
            return homography, mask, (xm, ym)
        else:
            print("Not Enough Matches. Matches Percentage is", pr)
            return None, None, None

    def constructMask(self, w, h, offset, expf=1.2):
        # Create an alpha blur on the left followed by white
        # using some exponential value to get better results
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        offset = int(offset)
        for i in range(0, offset):
            factor = np.clip((float(i) ** expf) / float(offset), 0.0, 1.0)
            c = int(factor * 255.0)
            # this is oddness in slice, need to submit bug report
            mask[0:h, i:i + 1, :] = (c, c, c)

        cv2.rectangle(mask, (w - offset, h), (offset, 0), color=(255, 255, 255), thickness=-1)
        return mask

    def buildPano(self, defished):
        # Build the panoram from the defisheye images
        offsets = []
        finalWidth = defished[0].width
        # Get the offsets and calculte the final size
        for i in range(0, len(defished) - 1):
            H, M, offset = self.findHomography(defished[i], defished[i + 1])
            dfw = defished[i + 1].width
            offsets.append(offset)
            finalWidth += int(dfw - offset[0])

        final = np.zeros((defished[0].height, finalWidth, 3), dtype=np.uint8)
        final[:defished[0].height, :defished[0].width, :] = defished[0]
        xs = 0
        # blit subsequent images into the final image
        for i in range(1, len(defished)):
            w = defished[i].width
            h = defished[i].height
            mask = self.constructMask(w, h, offsets[i - 1][0])

            xs += int(w - offsets[i - 1][0])
            cropped_defished = np.array(defished[i]) * 0.9 + mask * 0.1
            final[0: defished[i].height, xs: xs + defished[i].width, :] = cropped_defished

        return final