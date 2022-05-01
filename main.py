# from imgstitch.stitch_images import *
from Utils.MultiCameraStreamer import *


if __name__ == '__main__':

    camera_sources = [0]
    mc = MultiCameraStreamer(streaming_sources=camera_sources, apply_stitching=False, stitching_direction=1)
    mc.stream()

    #stitch_images_and_save("images", ["2.png", "4.png"], 1, "results")


