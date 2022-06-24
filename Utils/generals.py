import cv2


def find_cameras_indexes(count_looking_for=4):
    max_index = count_looking_for * 6
    cam_index = 0
    found_camera_indexes = []
    while cam_index < max_index and len(found_camera_indexes) < count_looking_for:
        cap = cv2.VideoCapture(cam_index)
        if cap.read()[0]:
            found_camera_indexes.append(cam_index)
            cap.release()
        cam_index += 1

    return found_camera_indexes
