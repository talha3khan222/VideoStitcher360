import cv2

from Utils.Camera import Camera
from imgstitch.utils import stitch_image_pair
from Utils.StitchImages import StitchImages
import multiprocessing
import concurrent.futures


current_frames = []


class MultiCameraStreamer:
    def __init__(self, streaming_sources=[], apply_stitching=True, stitching_direction=1):
        self._cameras = []
        for source in streaming_sources:
            self._cameras.append(Camera(source))
            current_frames.append(None)

        self._keep_streaming = True
        self._apply_stitching = apply_stitching
        self._stitching_direction = stitching_direction
        self._images_stitcher = StitchImages()

    def stream(self):
        '''for cam in self._cameras:
            ret_value = multiprocessing.Value("l", 0.0, lock=False)
            reader_process = multiprocessing.Process(target=cam.grab_frame, args=[ret_value])
            reader_process.start()
            reader_process.join()'''
        '''count = 0
        for cam in self._cameras:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(cam.grab_frame, 'world!')
                return_value = future.result()
                # print(return_value)
                current_frames[count] = return_value
                cv2.imshow(str(count), return_value)

            count += 1'''

        while self._keep_streaming:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(cam.grab_frame, "") for cam in self._cameras]
                frames = [f.result() for f in futures]

            for index, frame in enumerate(frames):
                if frame is not None:
                    # frame = cv2.resize(frame, (450, 300))
                    cv2.imshow(self._cameras[index].get_name(), frame)
                    cv2.imwrite('images/' + self._cameras[index].get_name() + ".png", frame)
                else:
                    print("Nothing")

            if self._apply_stitching:
                try:
                    stitched_frame = self._images_stitcher.stitch(frames[0], frames[1], self._stitching_direction)
                    if stitched_frame is not None:
                        pass
                        # cv2.imshow('stitched image', stitched_frame)
                except Exception as e:
                    print(e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._keep_streaming = False

        for cam in self._cameras:
            cam.release_handle()
        cv2.destroyAllWindows()

