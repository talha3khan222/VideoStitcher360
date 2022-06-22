# from imgstitch.stitch_images import *
from Utils.MultiCameraStreamer import *

from Utils.ParametersAdjuster import adjust_all_camera_parameters

if __name__ == '__main__':

    camera_sources = [0, 1, 2, 3]
    all_camera_parameters = adjust_all_camera_parameters(camera_sources)
    mc = MultiCameraStreamer(streaming_sources=camera_sources, apply_stitching=False, stitching_direction=1,
                             cam_parameters=all_camera_parameters)
    mc.stream()

    #stitch_images_and_save("images", ["2.png", "4.png"], 1, "results")


