import cv2
import numpy as np
from PIL import Image

from Utils.Camera import Camera
import concurrent.futures
from Utils.FeatureMatcher import FeatureMatcher
from Utils.generals import combine_images, trim
# from Utils.Tailor import Tailor


current_frames = []


'''def stitch(left, right):
    fm = FeatureMatcher()
    src_pts, dst_pts = fm.match_images(left, right)

    if src_pts is None:
        return

    MIN_MATCH_COUNT = 3
    if len(src_pts) <= MIN_MATCH_COUNT:
        print("Not enough matches found", len(src_pts), MIN_MATCH_COUNT)
        return None

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = left.shape[0], left.shape[1]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    # img2 = cv2.polylines(right, [np.int32(dst)], True, (0, 255, 255), 2, cv2.LINE_AA)
    # cv2.imshow("original_image_overlapping.jpg", img2)

    dst = cv2.warpPerspective(left, M, (right.shape[1] + left.shape[1], right.shape[0]))
    cv2.imshow('dst', dst)
    # cv2.waitKey()
    dst[0:right.shape[0], 0:right.shape[1]] = right
    cv2.imshow("original_image_stitched.jpg", dst)

    cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
    # cv2.waitKey()

    return trim(dst)'''


class MultiCameraStreamer:
    def __init__(self, streaming_sources=[], apply_stitching=True, cam_parameters=[]):
        self._cameras = []
        for cam_idx, source in enumerate(streaming_sources):
            self._cameras.append(Camera(source, doUnwarp=False, doCrop=False, parameters=cam_parameters[cam_idx]))
            current_frames.append(None)

        self._keep_streaming = True
        self._apply_stitching = apply_stitching

        # self._tailor = Tailor()
        self._stitcher = cv2.Stitcher_create()

        self._combinations = {}

        self._transformation_matrices = []
        self._transformation_matrices.append(np.array([[ 4.94666047e+00,  2.41170928e-01, -2.38876244e+03],
                                                       [ 1.03375334e+00,  4.54913411e+00, -9.31887226e+02],
                                                       [ 5.57908324e-03,  8.48213933e-04,  1.00000000e+00]]))

        self._transformation_matrices.append(np.array([[ 2.48237478e+00,  8.67674767e-02, -1.06397831e+03],
                                                       [ 4.26131976e-01,  1.98877399e+00, -1.58140696e+02],
                                                       [ 2.49989557e-03, -3.50564156e-04,  1.00000000e+00]]))

        self._transformations = []
        self._transformations.append(np.array([[0.96761879, -0.29169015, -135.20440722],
                                               [0.35775928, 0.94956671, -68.98758327],
                                               [0, 0, 1]]))

        self._transformations.append(np.array([[2.05597958e+00, -8.07220465e-01, -7.82495187e+02],
                                               [3.44762056e-01,  9.70113636e-01, -1.44046912e+02],
                                              [0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]))

    def stream(self):
        while self._keep_streaming:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(cam.grab_frame, "") for cam in self._cameras]
                frames = [f.result() for f in futures]

                h, w = frames[0].shape[0], frames[0].shape[1]

                for i in range(1, len(frames)):
                    frames[i] = cv2.resize(frames[i], (w, h))

            if not self._apply_stitching:
                for index, frame in enumerate(frames):
                    if frame is not None:
                        cv2.imshow(str(index), frame)
                        cv2.imwrite('images/' + str(index) + ".png", frame)
                    else:
                        print("Nothing", self._cameras[index].get_name())

            if self._apply_stitching:
                try:
                    # self.do_stitching(frames)

                    stitched12 = combine_images(frames[2], frames[1], np.linalg.inv(self._transformation_matrices[1]))
                    stitched01 = combine_images(stitched12, frames[0], np.linalg.inv(self._transformation_matrices[0]))

                    res = cv2.resize(stitched01, (1240, 480))

                    #cv2.imshow("Merged01", stitched01)
                    #cv2.imshow("Merged12", stitched12)
                    cv2.imshow("Res", res)

                    '''stitched23 = combine_images(frames[3], frames[2], self._transformation_matrices[2])
                    stitched12 = combine_images(frames[2], frames[1], self._transformation_matrices[1])
                    stitched01 = combine_images(frames[1], frames[0], self._transformation_matrices[0])

                    stitched123 = combine_images(stitched23, frames[1], self._transformation_matrices[0])
                    stitched0123 = combine_images(stitched123, frames[0], self._transformation_matrices[0])

                    cv2.imshow("Stitched01", stitched01)
                    cv2.imshow("Stitched23", stitched23)

                    cv2.imshow("Stitched012", stitched012)
                    cv2.imshow("Stitched0123", stitched0123)'''

                    # m1 = self.stitch_halves(frames[0], frames[2])
                    # m2 = cv2.hconcat((frames[1], frames[3]))

                    # cv2.imshow("Merged1", m1)
                    # cv2.imshow("Merged2", m2)

                    # match_and_stitch(frames[1], frames[0])

                    # cv2.imshow("012", stitched012)

                except Exception as e:
                    print(e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._keep_streaming = False

        for cam in self._cameras:
            cam.release_handle()
        cv2.destroyAllWindows()

    def stitch_halves(self, half1, half2):
        half_columns_count = round(half1.shape[1] / 2)
        q1, q2 = half1[:, :half_columns_count], half1[:, half_columns_count:]

        cv2.imwrite("images/q1.png", q1)
        cv2.imwrite("images/q2.png", q2)

        right_stitched = combine_images(q1, half2, np.linalg.inv(self._transformations[1]))
        stitched = combine_images(right_stitched, q2, np.linalg.inv(self._transformations[0]))

        # merged = cv2.hconcat((q2, half2, q1))

        return stitched

    def do_stitching(self, frames):
        stitched = None

        (status1, stitched1) = self._stitcher.stitch(frames)
        (status2, stitched2) = self._stitcher.stitch([frames[2], frames[3]])

        if status1 == 0:
            stitched1 = trim(stitched1)
            stitched1 = cv2.resize(stitched1, (480 * 2, 480))
            cv2.imwrite('Half1.png', stitched1)
            cv2.imshow('Half 1', stitched1)
        if status2 == 0:
            stitched2 = trim(stitched2)
            stitched2 = cv2.resize(stitched2, (480 * 2, 480))
            cv2.imwrite('Half2.png', stitched2)
            cv2.imshow('Half 2', stitched2)

        '''if status1 == 0 and status2 == 0:
            half_columns_count = round(stitched2.shape[1] / 2)

            stitched_halves = [stitched2[:, :half_columns_count],
                               stitched1,
                               stitched2[:, half_columns_count:]]

            (stitched_half_status1, stitched_half1) = self._stitcher.stitch(
                [stitched2[:, :half_columns_count], stitched1[:, half_columns_count:]])

            (stitched_half_status2, stitched_half2) = self._stitcher.stitch(
                [stitched1[:, :half_columns_count], stitched2[:, half_columns_count:]])

            if stitched_half_status1 == 0:
                stitched_half1 = trim(stitched_half1)
                stitched_half1 = cv2.resize(stitched_half1, (480 * 2, 480))
                cv2.imwrite('Stitched Half 1.png', stitched_half1)
                cv2.imshow('Stitched Half 1', stitched_half1)

            if stitched_half_status2 == 0:
                stitched_half2 = trim(stitched_half2)
                stitched_half2 = cv2.resize(stitched_half2, (480 * 2, 480))
                cv2.imwrite('Stitched Half 2.png', stitched_half2)
                cv2.imshow('Stitched Half 2', stitched_half2)

            # (status, stitched) = self._stitcher.stitch(stitched_halves)
            if stitched_half_status1 == 0 and stitched_half_status2 == 0:
                stitched = cv2.hconcat([stitched_half2, stitched_half1])
                stitched = cv2.resize(stitched, (480 * 2, 480))
                stitched = trim(stitched)
                cv2.imwrite('stitched.png', stitched)
                cv2.imshow('Stitched', stitched)'''

        return stitched


