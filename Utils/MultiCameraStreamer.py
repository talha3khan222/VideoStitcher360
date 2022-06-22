import cv2
import numpy as np
from PIL import Image

from Utils.Camera import Camera
from imgstitch.utils import stitch_image_pair
from Utils.StitchImages import StitchImages
import multiprocessing
import concurrent.futures
from Utils.PanoramaBuilder import PanoramaBuilder
from Utils.FeatureMatcher import FeatureMatcher


current_frames = []


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    #crop top
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def stitch(right, left):

    fm = FeatureMatcher()
    src_pts, dst_pts = fm.match_images(left, right)

    MIN_MATCH_COUNT = 2
    if src_pts is None or len(src_pts) < MIN_MATCH_COUNT:
        print("Minimum Matching Points: %d", MIN_MATCH_COUNT)
        return None

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = left.shape[0], left.shape[1]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(right,[np.int32(dst)], True, (0, 255, 255), 2, cv2.LINE_AA)
    # cv2.imshow("original_image_overlapping.jpg", img2)
    
    dst = cv2.warpPerspective(left, M, (right.shape[1] + left.shape[1], right.shape[0]))
    # cv2.imshow('dst', dst)
    # cv2.waitKey()
    dst[0:right.shape[0], 0:right.shape[1]] = right
    # cv2.imshow("original_image_stitched.jpg", dst)
    
    # cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
    return dst


class MultiCameraStreamer:
    def __init__(self, streaming_sources=[], apply_stitching=True, stitching_direction=1, cam_parameters=[]):
        self._cameras = []
        for cam_idx, source in enumerate(streaming_sources):
            self._cameras.append(Camera(source, doUnwarp=True, doCrop=False, parameters=cam_parameters[cam_idx]))
            current_frames.append(None)

        self._keep_streaming = True
        self._apply_stitching = apply_stitching
        self._stitching_direction = stitching_direction
        self._images_stitcher = StitchImages()
        
        self._pano_maker = PanoramaBuilder()

        self._stitcher = cv2.Stitcher_create()

    def stream(self):
        while self._keep_streaming:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(cam.grab_frame, "") for cam in self._cameras]
                frames = [f.result() for f in futures]

            if not self._apply_stitching:
                for index, frame in enumerate(frames):
                    if frame is not None:
                        resized_frame = cv2.resize(frame, (450, 300))
                        cv2.imshow(self._cameras[index].get_name(), frame)
                        cv2.imwrite('images/' + self._cameras[index].get_name() + ".png", frame)
                    else:
                        print("Nothing", self._cameras[index].get_name())

            if self._apply_stitching:
                '''stitched_frame = self._images_stitcher.stitch(frames[0], frames[1], self._stitching_direction)
                if stitched_frame is not None:
                    pass'''
                '''left = frames[1][:, :870]
                right = frames[3][:, 20:]

                # stitch(left, right)
                left = cv2.resize(left, (480, 480))
                right = cv2.resize(right, (480, 480))

                dst = np.concatenate((left, right), 1)
                cv2.imshow('stitched image', dst)'''
                '''final = cv2.resize(frames[0], (480, 480))
                percentage = 0.05
                percent_pixels = round(percentage * final.shape[1])
                final = final[:, percent_pixels:final.shape[1] - percent_pixels]
                
                for i in range(1, len(frames)):
                    # stitched_image = stitch(cv2.resize(frames[i], (480, 480)), cv2.resize(frames[i+1], (480, 480)))
                    left = final[:, :]
                    right = cv2.resize(frames[i], (480, 480))
                    
                    right = right[:, percent_pixels:right.shape[1] - percent_pixels]

                    final = np.concatenate((final, right), 1)
                
                # pano = self._pano_maker.buildPano([Image.fromarray(cv2.resize(defished, (480, 480))) for defished in frames])
                cv2.imshow('Pano', final)'''
                try:
                    (status, stitched) = self._stitcher.stitch(frames)
                    if status == 0:
                        cv2.imshow('Stitched', stitched)
                except Exception as e:
                    print(e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._keep_streaming = False

        for cam in self._cameras:
            cam.release_handle()
        cv2.destroyAllWindows()

