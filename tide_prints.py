
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import re

import cv2
import graph

def pick_points(im, picked_points_here):
    """
    Pick points on an image and collect the points into a list.  Coordinates are in pixels.
    Asynchronous function, so there is no return.  Results are appended to the list, which
    continues AFTER the function returns.
    """
    fig, ax = plt.subplots(1, 1, num=1)
    fig.tight_layout()
    ax.imshow(im)
    

    def onpress(event):
        picked_points_here.append((event.xdata, event.ydata))
        print('you pressed', event.button, event.xdata, event.ydata)

    fig.canvas.mpl_connect('button_press_event', onpress)


RUN_CODE = True

if RUN_CODE:
    # First let's display the boat and allow the user to click on the outline.
    DIR = '/Users/viola/canlogs/Currents'
    files = sorted([f for f in os.listdir(DIR) if re.match(r'.*\.png', f)])
    
    im = cv2.imread(os.path.join(DIR, files[0]))
    im = im[300:-300,200:-200,:]
    
    picked_points = []
    pick_points(im, picked_points)

    # Stored from a previous run
    picked_points = [(2.014617239300833, 246.14376130198923),
                     (3.431133212778832, 217.81344183242925),
                     (9.097197106690828, 189.48312236286927),
                     (23.26235684147082, 161.15280289330929),
                     (54.425708257986685, 139.9050632911393),
                     (102.58725135623865, 125.73990355635937),
                     (149.33227848101262, 117.24080771549137),
                     (203.1598854731766, 105.90867992766738),
                     (252.73794454490655, 98.82610006027738),
                     (296.6499397227245, 95.99306811332139),
                     (350.4775467148885, 90.32700421940939),
                     (392.97302591922835, 87.49397227245339),
                     (438.3015370705243, 88.91048824593139),
                     (482.2135322483423, 90.32700421940939),
                     (527.5420433996384, 91.74352019288739),
                     (571.4540385774563, 100.24261603375538),
                     (612.5330018083181, 105.90867992766738),
                     (663.527576853526, 115.82429174201337),
                     (708.856088004822, 127.15641952983736),
                     (751.351567209162, 139.9050632911393),
                     (793.847046413502, 152.6537070524413),
                     (832.092977697408, 168.23538276069928),
                     (874.5884569017479, 185.23357444243527),
                     (907.1683242917419, 196.56570223025926),
                     (949.6638034960819, 217.81344183242925),
                     (979.4106389391198, 229.14556962025324),
                     (1013.4070223025918, 246.14376130198923)]

    # Now lets fit a smooth curve and resample.
    points = np.array(picked_points)
    points = points / points.max()
    xm, ym = points.max(0)

    x_knots = points[:, 0]
    y_knots = ym - points[:, 1]

    # Very interesting, that this was "much" harder than I thought.  The naive way to fit a
    # spline to a shape is as a function (the y values are a function of the x values).  But
    # there are many shapes for which this is not a great fit (because the data does not look
    # like a function... derivative is infinite).
    #
    # So I screwed around with doing it in the as polar function (below).  But that is crazy.

    x_knots = points[:, 0]
    y_knots = ym - points[:, 1]
    # both are implicit functions of t
    t_knots = np.arange(len(x_knots))

    fx = scipy.interpolate.InterpolatedUnivariateSpline(t_knots, x_knots)
    fy = scipy.interpolate.InterpolatedUnivariateSpline(t_knots, y_knots)

    # Now resample and close the curve
    t_vals = np.linspace(0, t_knots.max(), num=7)

    x_vals = fx(t_vals)
    y_vals = fy(t_vals)
    x_vals[1] = x_vals[0]
    x_vals = np.hstack((x_vals, np.flip(x_vals)))
    y_vals = np.hstack((y_vals, -np.flip(y_vals)))

    fig, ax = graph.new_axis(2, True, True)
    ax.plot(x_knots, y_knots, linestyle = 'None', marker='.')
    ax.plot(x_vals, y_vals, marker='.')

    OUTLINE = np.vstack((y_vals, x_vals)).T
    OUTLINE = OUTLINE - OUTLINE.mean(0)
    OUTLINE = OUTLINE / (x_vals.max() - x_vals.min(), 1)

    fig, ax = graph.new_axis(3, True, True)
    ax.plot(OUTLINE[:, 0], OUTLINE[:, 1], marker='.')


    ################################################################
    # Weird polar approach.
    #
    # squish = 1/6.0
    # x_knots = squish * (x_knots - x_knots.mean())
    # y_knots = y_knots - y_knots.mean()

    # im_knots = 1j * x_knots + y_knots

    # a_knots = np.angle(im_knots)
    # l_knots = np.abs(im_knots)

    # # f = scipy.interpolate.CubicSpline(x_knots, y_knots)
    # # f = scipy.interpolate.UnivariateSpline(x_knots, y_knots, s=0.001)

    # f = scipy.interpolate.interp1d(a_knots, l_knots, kind='linear')
    # f = scipy.interpolate.Rbf(a_knots, l_knots, smooth=0.1)

    # a_vals = np.linspace(a_knots.min(), a_knots.max(), num=10)
    # l_vals = f(a_vals)

    # im_vals = l_vals * np.exp(1j * a_vals)
    # x_vals = np.imag(im_vals)
    # y_vals = np.real(im_vals)

    # fig, ax = graph.new_axis(2, True, True)
    # ax.plot(x_knots/squish, y_knots, linestyle = 'None', marker='.')
    # ax.plot(x_vals/squish, y_vals, marker='.')
