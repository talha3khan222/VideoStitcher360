from Utils.MultiCameraStreamer import *
from Utils.ParametersAdjuster import adjust_all_camera_parameters
import pickle
from Utils.generals import find_cameras_indexes


if __name__ == '__main__':

    cameras_count = 4
    # camera_sources = find_cameras_indexes(cameras_count)
    # print(camera_sources)
    camera_sources = [0, 1, 2, 3]
    if len(camera_sources) < cameras_count:
        print("Couldn't find the required number of cameras. Required Cameras Count was:", cameras_count)
    else:
        load_old_parameters = True
        parameters_file_path = "parameters.pickle"

        if not load_old_parameters:
            all_camera_parameters = adjust_all_camera_parameters(camera_sources)
            with open(parameters_file_path, 'wb') as pf:
                pickle.dump(all_camera_parameters, pf)

        with open(parameters_file_path, 'rb') as pf:
            all_camera_parameters = pickle.load(pf)

        mc = MultiCameraStreamer(streaming_sources=camera_sources,
                                 apply_stitching=True,
                                 stitching_direction=1,
                                 cam_parameters=all_camera_parameters)
        mc.stream()

