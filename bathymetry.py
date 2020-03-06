

import numpy as np
import os
import cv2
import utils as u
import skimage
import ocv

import globals as G
import proces as p


def create_bag(longitudes, latitudes, border=0.2):
    border = 2.0

    longitudes = np.asarray(longitudes)
    latitudes  = np.asarray(latitudes)

    # Add/sub just a bit to make the map interpretable for small excursions
    lat_max, lat_min, lat_mid = p.max_min_mid(latitudes, border) + (0.002, -0.002, 0.0)
    lon_max, lon_min, lon_mid = p.max_min_mid(longitudes, border) + (0.002, -0.002, 0.0)

    # Find the extents of the map
    west, south = np.array(G.MAP(lon_min, lat_min))
    east, north = np.array(G.MAP(lon_max, lat_max))

    # Setup gdalwarp args
    # Define the extent of the map in lon/lat
    te = f"-te {lon_min:.7f} {lat_min:.7f} {lon_max:.7f} {lat_max:.7f}"
    t_srs = f"-t_srs '{G.PROJ4}'"
    # Size of the image should match the shape of the map
    if (east - west) > (north - south):
        ts = "-ts 2000 0"
    else:
        ts = "-ts 0 2000"
    chart_file = os.path.join('/tmp/bag8.tif')

    zoom = "-oo ZOOM_LEVEL=16"
    zoom = ""
    command1 = f"gdalwarp {zoom} {te} {t_srs} -te_srs EPSG:4326 -r bilinear"
    command1 += " " + ts
    # command1 += " " + os.path.join('/Users/viola/canlogs/BAG/H12024_MBVB_1m_MLLW_1of4.bag')
    command1 += " " + os.path.join('/Users/viola/canlogs/BAG/big.bag')
    command1 += " " + chart_file

    print(command1)
    os.system(f"rm {chart_file}")
    os.system(command1)
    image = skimage.io.imread(chart_file)[:,:,0]
    image[image > 0] = 0

    ocv.grayshow(image, 8)

    track = np.vstack(G.MAP(longitudes, latitudes)).T - (west, south)

    return u.DictClass(image=image, west=west, east=east, north=north, south=south,
                       track=track)

def prepare():

    ttt = skimage.io.imread("/Users/viola/canlogs/BAG/tmp.tif")
    ttt = skimage.io.imread("/Users/viola/canlogs/BAG/big_8m.tif")
    ttta = ttt[:,:,0]
    tttb = ttt[:,:,1]
    ttta[ttta > 0] = 0
    ocv.grayshow(ttta, 1)
    ocv.grayshow(tttb, 1)

    imb = skimage.io.imread("/Users/viola/canlogs/BAG/bag.tif")
    img = skimage.io.imread("/Users/viola/canlogs/BAG/gebco.tif")

    him = imb[:,:,0]
    cim = imb[:,:,1]
    ocv.grayshow(him, 1)

    kernel = np.ones((7, 7), np.uint8)
    depth_near = cv2.dilate(him, kernel, iterations=3)
    mask = np.logical_and(depth_near < -10, him == 0.0).astype(np.uint8)

    ocv.grayshow(depth_near, 3)
    img[img > 10] = 10
    
    # mask = (him==0.0).astype(np.uint8)
    shallow = np.logical_and(him > -3, him <= 0.0).astype(np.uint8)
    shallow = cv2.dilate(shallow, kernel, iterations=5)
    ocv.grayshow(shallow, 3)

    mask = (him == 0.0).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    dmask = cv2.dilate(mask, kernel, 3)

    him[mask == 1] = img[mask == 1]

    ocv.grayshow(him, 2)
    highlight = cv2.Sobel(him, cv2.CV_32F, 1, 0, ksize=5) + cv2.Sobel(him, cv2.CV_32F, 0, 1, ksize=5)
    ocv.grayshow(highlight, 3)
