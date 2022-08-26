import cv2
import numpy as np
from PIL import Image

from Utils.Camera import Camera
import concurrent.futures
from Utils.FeatureMatcher import FeatureMatcher
from Utils.generals import trim, stitch


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
            self._cameras.append(Camera(source, doUnwarp=True, doCrop=False, parameters=cam_parameters[cam_idx]))
            current_frames.append(None)

        self._keep_streaming = True
        self._apply_stitching = apply_stitching

        self._transformation_matrices = []
        self._transformation_matrices.append(np.array([[ 8.69254188e-01, -1.20607271e-01,  2.46391392e+02],
                                                       [-5.89547012e-02,  8.49158430e-01,  5.96098177e+00],
                                                       [-5.30238586e-05, -3.77897860e-04,  1.00000000e+00]]))

        self._transformation_matrices.append(np.array([[ 1.12078614e+00,  1.81963876e-01,  1.77567230e+02],
                                                       [-7.91934894e-02,  1.08327569e+00,  5.99232166e+00],
                                                       [ 1.20473453e-04,  1.09706471e-04,  1.00000000e+00]]))

    def stream(self):
        while self._keep_streaming:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(cam.grab_frame, "") for cam in self._cameras]
                frames = [f.result() for f in futures]

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
                    stitched12 = stitch(frames[2], frames[1], self._transformation_matrices[1])
                    stitched = stitch(stitched12, frames[0], self._transformation_matrices[0])
                    cv2.imshow("Stitched", stitched)
                    cv2.imwrite("stitched012.png", stitched)

                except Exception as e:
                    print(e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._keep_streaming = False

        for cam in self._cameras:
            cam.release_handle()
        cv2.destroyAllWindows()

    '''def do_stitching(self, frames):
        stitched = None

        (status1, stitched1) = self._stitcher.stitch([frames[0], frames[1]])
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

        if status1 == 0 and status2 == 0:
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
                cv2.imshow('Stitched', stitched)

        return stitched'''


