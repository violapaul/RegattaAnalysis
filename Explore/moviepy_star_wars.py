"""
Description of the video:
Mimic of Star Wars' opening title. A text with a (false)
perspective effect goes towards the end of space, on a
background made of stars. Slight fading effect on the text.

"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import moviepy
from moviepy.editor import TextClip, ImageClip, CompositeVideoClip
from moviepy.video.tools.drawing import color_gradient
from skimage import transform as tf
import logging

moviepy.config.change_settings(dict(FFMPEG_BINARY="/Users/viola/Bin/ffmpeg"))

# RESOLUTION

stars = ImageClip('Data/Images/movie_grab_01.png')
w, h = stars.size
moviesize = w, h


# THE RAW TEXT
raw_txt = "\n".join([
    "A long time ago, in a faraway galaxy,",
    "there lived a prince and a princess",
    "who had never seen the stars, for they",
    "lived deep underground.",
    "",
    "Many years before, the prince's",
    "grandfather had ventured out to the",
    "surface and had been burnt to ashes by",
    "solar winds.",
    "",
    "One day, as the princess was coding",
    "and the prince was shopping online, a",
    "meteor landed just a few megameters",
    "from the couple's flat."
])

# Add blanks
lines = 10
txt = lines * "\n" + raw_txt + (30-lines)*"\n"

# CREATE THE TEXT IMAGE
clip_txt = TextClip(txt, color='white', align='West', fontsize=25,
                    font='Xolonium-Bold', method='label')

# SCROLL THE TEXT IMAGE BY CROPPING A MOVING AREA

txt_speed = 27

def fl(gf, t):
    logging.warning(f"{type(gf)} {t}")
    return gf(t)[int(txt_speed*t):int(txt_speed*t)+h, :]


moving_txt = clip_txt.fl(fl, apply_to=['mask'])

# ADD A VANISHING EFFECT ON THE TEXT WITH A GRADIENT MASK

grad = color_gradient(moving_txt.size, p1=(0, int(2*h/3)),
                      p2=(0, int(h/4)), col1=0.0, col2=1.0)
gradmask = ImageClip(grad, ismask=True)


def fl_grad_mask(pic):
    # logging.warning(f"foobar {pic.shape}, {gradmask.img.shape}")
    return np.minimum(pic, gradmask.img)

moving_txt.mask = moving_txt.mask.fl_image(fl_grad_mask)

# WARP THE TEXT INTO A TRAPEZOID (PERSPECTIVE EFFECT)
def trapzWarp(pic, cx, cy, ismask=False):
    """ Complicated function (will be latex packaged as a fx) """
    Y, X = pic.shape[:2]
    src = np.array([[0,0], [X,0], [X,Y], [0,Y]])
    dst = np.array([[cx*X,cy*Y], [(1-cx)*X,cy*Y], [X,Y], [0,Y]])
    tform = tf.ProjectiveTransform()
    tform.estimate(src, dst)
    im = tf.warp(pic, tform.inverse, output_shape = (Y, X))
    logging.warn(f"{Y}, {X}, {pic.max()} {im.max()}")
    return im if ismask else (im*255).astype('uint8')

def fl_im(pic):
    return trapzWarp(pic, 0.2, 0.3)

def fl_mask(pic):
    return trapzWarp(pic, 0.2, 0.3, ismask=True)


warped_txt= moving_txt.fl_image(fl_im)
warped_txt.mask = warped_txt.mask.fl_image(fl_mask)


# BACKGROUND IMAGE, DARKENED AT 60%

stars_darkened = stars.fl_image(lambda pic: (0.6*pic).astype('int16'))


# COMPOSE THE MOVIE

final = CompositeVideoClip([
    stars_darkened,
    warped_txt.set_pos(('center','bottom'))],
                           size = moviesize)


final.show(1.5, interactive = True)


import time
tic = time.perf_counter()
final.set_duration(8).write_videofile("starworms.mp4", fps=30, codec='hevc_videotoolbox', logger=None)
toc = time.perf_counter()
print(toc-tic)


import cProfile

# WRITE TO A FILE

cProfile.run('final.set_duration(8).write_videofile("starworms.mp4", codec="hevc_videotoolbox", fps=30, logger=None)', 'profile')
import pstats

p = pstats.Stats('profile')
p.strip_dirs().sort_stats(-1).print_stats()

p.sort_stats('time').print_stats(20)
p.sort_stats('cumulative').print_stats()
