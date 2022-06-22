import cv2
from PIL import Image
import numpy as np

print("Hello")


def equirect_proj(x_proj, y_proj, W, H, fov):
    """Return the equirectangular projection on a unit sphere,
    given cartesian coordinates of the de-warped image."""
    theta_alt = x_proj * fov / W
    phi_alt = y_proj * np.pi / H

    x = np.sin(theta_alt) * np.cos(phi_alt)
    y = np.sin(phi_alt)
    z = np.cos(theta_alt) * np.cos(phi_alt)

    return np.arctan2(y, x), np.arctan2(np.sqrt(x**2 + y**2), z)


class DeFishEye:
    def __init__(self, field_of_view_h=160, field_of_view_v=160):
        self._mapping_x = None
        self._mapping_y = None
        self._hfov = field_of_view_h
        self._vfov = field_of_view_v
        
    def buildmap(self, Ws, Hs, Wd, Hd, fov=195):
        """Return a mapping from de-warped images to fisheye images."""
        fov = fov * np.pi / 180.0

        # cartesian coordinates of the de-warped rectangular image
        ys, xs = np.indices((Hs, Ws), np.float32)
        y_proj = Hs / 2.0 - ys
        x_proj = xs - Ws / 2.0

        # spherical coordinates
        theta, phi = equirect_proj(x_proj, y_proj, Ws, Hs, fov)

        # polar coordinates (of the fisheye image)
        p = Hd * phi / fov

        # cartesian coordinates of the fisheye image
        y_fish = p * np.sin(theta)
        x_fish = p * np.cos(theta)

        ymap = Hd / 2.0 - y_fish
        xmap = Wd / 2.0 + x_fish
        
        self._mapping_x = xmap
        self._mapping_y = ymap
        # return xmap, ymap

    def buildMap(self, Ws, Hs, Wd, Hd):
        # Build the fisheye mapping
        map_x = np.zeros((Hd, Wd), np.float32)
        map_y = np.zeros((Hd, Wd), np.float32)
        vfov = (self._vfov / 180.0) * np.pi
        hfov = (self._hfov / 180.0) * np.pi
        vstart = ((180.0 - self._vfov) / 180.00) * np.pi / 2.0
        hstart = ((180.0 - self._hfov) / 180.00) * np.pi / 2.0
        count = 0
        # need to scale to changed range from our
        # smaller cirlce traced by the fov
        xmax = np.sin(np.pi / 2.0) * np.cos(vstart)
        xmin = np.sin(np.pi / 2.0) * np.cos(vstart + vfov)
        xscale = xmax - xmin
        xoff = xscale / 2.0
        zmax = np.cos(hstart)
        zmin = np.cos(hfov + hstart)
        zscale = zmax - zmin
        zoff = zscale / 2.0
        # Fill in the map, this is slow but
        # we could probably speed it up
        # since we only calc it once, whatever
        for y in range(0, int(Hd)):
            for x in range(0, int(Wd)):
                count = count + 1
                phi = vstart + (vfov * ((float(x) / float(Wd))))
                theta = hstart + (hfov * ((float(y) / float(Hd))))
                xp = ((np.sin(theta) * np.cos(phi)) + xoff) / zscale  #
                zp = ((np.cos(theta)) + zoff) / zscale  #
                xS = Ws - (xp * Ws)
                yS = Hs - (zp * Hs)
                map_x.itemset((y, x), int(xS))
                map_y.itemset((y, x), int(yS))

        self._mapping_x = map_x
        self._mapping_y = map_y

    def unwarp(self, img, doCrop=False):
        # apply the unwarping map to our image
        output = cv2.remap(np.array(img), self._mapping_x, self._mapping_y, cv2.INTER_NEAREST)
        # result = np.array(output, dtype=np.uint8)
        if doCrop:
            output = np.array(self.postCrop(Image.fromarray(output)))
        return output

    def postCrop(self, img, threshold=10):
        # Crop the image after dewarping
        # return img.crop((img.width * 0.28, img.height * 0.01, img.width * 0.8, img.height * 0.99))
        return img.crop((img.width * 0.15, img.height * 0.01, img.width * 0.85, img.height * 0.99))
