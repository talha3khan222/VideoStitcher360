from Utils.MultiCameraStreamer import *
from Utils.ParametersAdjuster import adjust_all_camera_parameters
import pickle


if __name__ == '__main__':

    camera_sources = [0, 1, 2, 3]
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



