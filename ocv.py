# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:19:49 2015

Helper functions related to OpenCV,  or ocv for short.

- Tools to stream images in various ways (lazy generators)
- Tools to show images
- Tools to proces and draw on images

@author: pviola

"""


import math
import numpy as np
import itertools as it
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
import re
import yaml


def imsize(opencv_image_as_numpy_array):
    '''Computes the size of an image for consumption by opencv functions.'''
    shape = opencv_image_as_numpy_array.shape
    height = shape[0]
    width  = shape[1]
    return (width, height)

################################################################
# Reads the broken YAML files from OpenCV.

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat

yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

def read_yaml(filepath):
    """Read an OpenCV YAML file."""
    with open(filepath) as fin:
        result = yaml.load(fin.read())
    return result


################################################################
# image viewers / visualize

def plt_hist(vals, figure=1, numBins=10):
    fig = plt.figure(figure)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.hist(vals,numBins,color='green',alpha=0.8)
    plt.show()    

def impixelinfo(ax=None, image=None):
    """Mimic Matlab's `impixelinfo` function that shows the image pixel
    information as the cursor swipes through the figure.

    Parameters
    ----------
    ax: axes
        The axes that tracks cursor movement and prints pixel information.
        We require the `ax.images` list to be non-empty, and if more than one
        images present in that list, we examine the last (newest) one. If not
        specified, default to 'plt.gca()'.

    image: ndarray
        If specified, use this `image`'s pixel instead of `ax.images[-1]`'s.
        The replacement `image` must have the same dimension as `ax.images[-1]`,
        and we will still be using the `extent` of the latter when tracking
        cursor movement.

    Returns
    -------
    None

    """
    # Set default 'ax' to 'plt.gca()'.
    if ax is None:
        ax = plt.gca()
    # Examine the number of images in 'ax'.
    if len(ax.images) == 0:
        print("No image in axes to visualize.")
        return
    # Set default 'image' if not specified.
    if image is None:
        image = ax.images[-1].get_array()
    # Get the 'extent' of current image.
    (left,right,bottom,top) = ax.images[-1].get_extent()

    # Re-define the 'format_coord' function and assign it to 'ax'.
    def format_coord(x, y):
        """Return a string formatting the `x`, `y` coordinates, plus additional
        image pixel information."""
        result_str = "(%.3f, %.3f): " % (x, y)
        # Get the image pixel index.
        i = int(math.floor((y - top) / (bottom - top) * image.shape[0]))
        j = int(math.floor((x - left) / (right - left) * image.shape[1]))
        # Return early if (i,j) is out of boundary.
        if (i < 0) or (i >= image.shape[0]) or (j < 0) or (j >= image.shape[1]):
            return result_str
        # Get the pixel value and add to return string.
        if (len(image.shape) == 3) and (image.shape[2] == 4):
            # 4-channel RGBA image.
            result_str += "(%.3f, %.3f, %.3f, %.3f)" % \
                          (image[i,j,0], image[i,j,1],
                           image[i,j,2], image[i,j,3])
        elif (len(image.shape) == 3) and (image.shape[2] == 3):
            # 3-channel RGB image.
            result_str += "(%.3f, %.3f, %.3f)" % \
                          (image[i,j,0], image[i,j,1], image[i,j,2])
        else:
            # Single-channel grayscale image.
            assert len(image.shape) == 2
            result_str += "%.3f" % image[i,j]
        return result_str
    ax.format_coord = format_coord


def multi_imshow(imlist, fignum=1, spread=False, colormaps=None, vertical=False, block=False):
    count = len(imlist)
    if vertical:
        rows, cols = count, 1
    else:
        rows, cols = 1, count
    fig = plt.figure(fignum)
    fig.clf()
    if colormaps is None:
        colormaps = ['gray' for im in imlist]
    ax_base = None
    for im, cmap, num  in zip(imlist, colormaps, range(1, count+1)):
        if ax_base is None:
            ax = plt.subplot(rows, cols, num)
            ax_base = ax
        else:
            ax = plt.subplot(rows, cols, num, sharex=ax, sharey=ax)
        img_plot = ax.imshow(im, cmap, interpolation='nearest')
        impixelinfo(ax)
        if spread:
            img_plot.set_clim(im.min(), im.max())
            img_plot.figure.canvas.draw()
        plt.colorbar(ax.images[-1])
    plt.tight_layout()

def multi_plot(plots, fignum=1, vertical=True):
    count = len(plots)
    if vertical:
        rows, cols = count, 1
    else:
        rows, cols = 1, count
    fig = plt.figure(fignum)
    fig.clf()
    ax_base = None
    for data, num  in zip(plots, range(1, count+1)):
        if ax_base is None:
            ax = plt.subplot(rows, cols, num)
            ax_base = ax
        else:
            if vertical:
                ax = plt.subplot(rows, cols, num, sharex=ax)
            else:
                ax = plt.subplot(rows, cols, num, sharey=ax)
        ax.plot(data)


    
def plt_show(im, num=1, spread=False, colormap='gray', block=True):
    fig = plt.figure(num)
    fig.clf()
    ax = fig.gca()
    img_plot = ax.imshow(im, colormap, interpolation='nearest')
    if spread:
        img_plot.set_clim(im.min(), im.max())
        img_plot.figure.canvas.draw()
    # plt.show(block=block)
    plt.colorbar(ax.images[-1])
    plt.pause(0.05)
    return ax

def grayshow(im, num):
    ax = plt_show(im, num)
    impixelinfo(ax)
    return ax

def jetshow(im, num):
    ax =  plt_show(im, num, colormap='jet')
    impixelinfo(ax)
    return ax

def plt_images(images, spread=False, colormap='gray'):
    for img in images:
        if type(img) is np.ndarray:
            plt_show(img, spread=spread, colormap=colormap, block=True)

def plt_zoom_window(fig=None):
    if fig is None:
        fig = plt.gcf()
    try:
        ax = fig.canvas.figure.get_axes()[0]
        return (sorted(ax.get_xlim()), sorted(ax.get_ylim()))
    except Exception as e:
        print(e.message)
        print("Could not retrieve zoom window.")

def image_show(img, destroy=True):
    """Blocking display of an image using cv2.  Any key to exit."""
    cv2.imshow('image', img)
    key = cv2.waitKey()
    if destroy:
        cv2.destroyAllWindows()
    return key

def image_spread(im, out_min=0, out_max=1):
    in_min = im.min()
    in_max = im.max()
    r = (out_max - out_min) / (in_max - in_min)
    return r * im + (out_min - r * in_min)

def display_images(images, spread=False):
    """Display a sequence of images. Hit any key to cycle.  q to quit."""
    cont = True
    for img in images:
        if not cont:
            break
        if spread:
            cv2.imshow('frame', image_spread(img, 0, 1))
        else:
            try:
                cv2.imshow('frame', img)
            except:
                pass
        while cont:
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cont = False
                break
            if key != -1:
                break
    cv2.destroyAllWindows()


    

def imflip(*images):
    """Flip through a set of images.  Good way to see changes and small motions. 
    Hit any key to cycle.  q to quit."""
    display_images(it.cycle(images))


def color_set(count, cmap=mpl.cm.jet):
    """return a set of colors that can be used to show COUNT things."""
    return map(lambda n: cmap(n / (1 + float(count))), xrange(count))

def plt_color_to_ocv(color_vec):
    cv = color_vec[0:3]
    return map(lambda x: int(x*255), cv)

################################################################
# Image operations

def image_reduce(img, steps):
    """reduce an image by n steps, using pyramid."""
    for i in range(steps):
        img = cv2.pyrDown(img)
    return img

def image_row_tile(img_list, width):
    """Given stream of images, produce a stream of rows which are WIDTH tiles wide."""
    seq = [iter(img_list)] *  width
    for group in it.izip(*seq):
        yield np.hstack(group)

def image_col_tile(img_list, height):
    """Given stream of images, produce a stream of cols which are HEIGHT tiles high."""
    seq = [iter(img_list)] *  height
    for group in it.izip(*seq):
        yield np.vstack(group)

def image_grid_tile(img_list, width, max_height=20):
    """Given stream of images, produce a grid which is WIDTH tiles wide (and MAX_HEIGHT high)."""
    rows = list(it.islice(image_row_tile(img_list, width), 0, max_height))
    if len(rows) > 0:
        return np.vstack(rows)
    else:
        return None

def pyramid_image(img, steps):
    img2 = img.copy()
    img2[:,:,:] = 0
    small = img
    offset = 0
    for i in range(steps):
        small = cv2.pyrDown(small)
        h, w, d = small.shape
        img2[0:h, offset:offset+w, :] = small
        offset = offset+w
    return np.hstack((img, img2))

def pyramid(img, steps):
    small = img
    yield img
    for i in range(steps):
        small = cv2.pyrDown(small)
        yield small

def image_reduce_to_max(img, max_dim):
    """Repeated downsample by 2x until the max(w, h) < max_dim."""
    h, w = img.shape[:2]
    if (h > max_dim) or (w > max_dim):
        return image_reduce_to_max(cv2.pyrDown(img), max_dim)
    else:
        return img

def rgb_color(img):
    if len(img.shape) == 3:
        return img.copy()
    else:
        try:
            flag = cv2.cv.CV_GRAY2RGB
        except:
            flag = cv2.COLOR_GRAY2RGB
    return cv2.cvtColor(img, flag)
    
def draw_horizontal_lines(img, count=10, color=(0, 255, 255)):
    h, w = img.shape[:2]
    delta = h/count
    cimg = rgb_color(img)
    for y in range(delta, h-delta, delta):
        cv2.line(cimg, (0, y), (w-1, y), color)
    return cimg

def draw_vertical_lines(img, count=10, color=(0, 255, 255)):
    h, w = img.shape[:2]
    delta = w/count
    cimg = rgb_color(img)
    for x in range(delta, w, delta):
        cv2.line(cimg, (x, 0), (x, h-1), color)
    return cimg

def draw_points(img, pts, radius=4, thickness=2, color=(0, 0, 255), ocv_coords=True):
    cimg = rgb_color(img)
    for pt in pts:
        if ocv_coords:
            y, x = pt
        else:
            x, y = pt
        cv2.circle(cimg, (int(round(y)), int(round(x))), radius, color, thickness)
    return cimg

def draw_tracks(img, kps1, kps2, radius=4, thickness=2, ocv_coords=True):
    cimg = rgb_color(img)
    for (p1, p2) in it.izip(kps1, kps2):
        # rows and cols
        if ocv_coords:
            y1, x1 = p1
            y2, x2 = p2
        else:
            x1, y1 = p1
            x2, y2 = p2
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        cv2.circle(cimg, (y1, x1), radius, (0, 0, 255), thickness)
        cv2.circle(cimg, (y2, x2), radius, (255, 0, 0), thickness)
        cv2.line(cimg, (y1, x1), (y2, x2), (255, 255, 0), thickness)
    return cimg


################################################################
# streaming image tools
#
# Several of the functions below uses iterables (by way of yield),
# both for flexibility, and for efficiency.  These functions take and
# return a stream of results.

def read_images(f1):
    """Read a stream of images from the videocapture device."""
    cap1 = cv2.VideoCapture(f1)
    cont = True
    while(cont and cap1.isOpened()):
        ret1, frame1 = cap1.read()

        if not (ret1):
            break

        yield frame1
    cap1.release()

def read_stereo(f1, f2):
    """zip together two image streams"""
    return it.izip(read_images(f1), read_images(f2))


def read_images_directory(directory, pattern=re.compile(".*")):
    """Read a stream of images from the files in a directory that match a pattern."""
    for (dirpath, dirname, filenames) in os.walk(directory):
        for file in filenames:
            if pattern.match(file):
                yield cv2.imread(os.path.join(dirpath, file))

def downsample_stereo(stereo_pairs, max_size=640):
    for (img_l, img_r) in stereo_pairs:
        yield (image_reduce_to_max(img_l, max_size), image_reduce_to_max(img_r, max_size))


def combine_images(image_tuples):
    """Combine images into a single image, horizontally.  Assumes all are the same height."""
    for tuple in image_tuples:
        yield np.hstack(tuple)


def show_stereo_video(stereo_stream, max_size=640):
    # uses streaming so this does not require you to read all the
    # images before processing or displaying.
    display_images(
        combine_images(
            downsample_stereo( stereo_stream, max_size)))


def show_video(image_stream, max_size=640):
    display_images(
        it.imap(lambda img: image_reduce_to_max(img, max_size), image_stream))


################################################################

def find_symbol(match):
    import re
    r = re.compile(match, re.IGNORECASE)
    mm = []
    for s in dir(cv2):
        if r.search(s):
            mm.append('cv2.' + s)
    try:
        for s in dir(cv2.cv):
            if r.search(s):
                mm.append('cv2.cv.' + s)
    except (NameError, AttributeError):
        pass
    return mm

################################################################

def animated_gif(gif_file, *images):
    '''Create an animated GIF from a set of images.  Assumes you have imagemagick installed.'''
    import tempfile
    import shutil
    import os
    import subprocess

    cmd = 'convert -delay 35 -loop 0'.split()
    dirpath = tempfile.mkdtemp()
    num = 0
    files = []
    for im in images:
        file = os.path.join(dirpath, "im%06d.png" % num)
        cv2.imwrite(os.path.join(file), im)
        files.append(file)
        num += 1
    subprocess.call(cmd + files + [gif_file])
    shutil.rmtree(dirpath)    
