import cv2

from Camera import Camera
import multiprocessing
import concurrent.futures


current_frames = []


class MultiCameraStreamer:
    def __init__(self, streaming_sources=[]):
        self._cameras = []
        for source in streaming_sources:
            self._cameras.append(Camera(source))
            current_frames.append(None)

        self._keep_streaming = True

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
                    frame = cv2.resize(frame, (450, 300))
                    cv2.imshow(self._cameras[index].get_name(), frame)
                    cv2.imwrite('images/' + self._cameras[index].get_name() + ".png", frame)
                else:
                    print("Nothing")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._keep_streaming = False

        for cam in self._cameras:
            cam.release_handle()
        cv2.destroyAllWindows()

