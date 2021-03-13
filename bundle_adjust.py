
import sys
import numpy as np
from numpy.linalg import norm
import math
import copy
from matplotlib import pyplot as plt

import pyqtgraph
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from PyQt4 import QtGui


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)
#     import ipdb; ipdb.set_trace()

################
# These imports are used for debuging
sys.path.insert(0, '/Users/pviola/Src/AdnCv/libraries')

import roslog

roslog.fast_setup("DEBUG")

def small_angle(angle):
    """Deals with the 2pi ambiguity of angles."""
    a = math.fmod(angle, 2 * math.pi)
    if a > math.pi:
        return a - (2 * math.pi)
    else:
        return a

################################################################
# Basic code for manipulating matrices, rotations, quarternions, etc.
def cross_mat(v):
    "Convert a vector into a matrix, which when multiplied by another vector computes the cross product."
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def q_vw(q):
    """Break a quaternion into 'axis' part and angle part."""
    return q[:3], q[3]

def vw_q_internal(v, w):
    """
    Create a quaternion from a an axis, and angle.  Do not use this, it is intended to
    hide the internal representation of the quaternion.
    """
    return np.hstack([v, w])

def xyzw_q(x, y, z, w):
    return np.array([x, y, z, w])

def q_identity():
    return vw_q_internal([0, 0, 0], 1.0)

def axis_angle_q(axis, angle):
    "Equation 2.39 in Hartley and Zisserman."
    norm_axis = np.array(axis) / norm(axis)
    return vw_q_internal(math.sin(angle/2) * norm_axis, math.cos(angle/2))

def axis_angle_twist(axis, angle):
    return (angle/norm(axis)) * np.array(axis)

def twist_q(twist):
    angle = norm(twist)
    return axis_angle_q(twist / angle, angle)

def q_twist(q):
    v, w = q_vw(q)
    n = norm(v)
    angle = 2 * math.atan2(n, w)
    return (angle/n) * v

def q_angle(q):
    v, w = q_vw(q)
    n = norm(v)
    return 2 * math.atan2(n, w)

def q_inverse(q):
    v, w = q_vw(q)
    return vw_q_internal(v, -w)

def q_compose(q0, q1):
    "Equation 2.42."
    v0, w0 = q_vw(q0)
    v1, w1 = q_vw(q1)
    return vw_q_internal(np.cross(v0, v1) + w0*v1 + w1*v0, w0*w1 - np.dot(v0, v1))

def q_mat(q):
    "Equation 2.40."
    v, w = q_vw(q)
    cx = cross_mat(v)
    return np.identity(3) + 2 * w * cx + 2 * np.dot(cx, cx)

def twist_mat(twist):
    "Rodriguez or Equation 2.34."
    vtwist = np.array(twist)
    angle = norm(vtwist)
    if math.fabs(angle) > 0:
        normal_vector = twist / angle
        cx = cross_mat(normal_vector)
        return np.identity(3) + math.sin(angle) * cx + (1 - math.cos(angle)) * np.dot(cx, cx)
    else:
        return np.identity(3)

def twist_inverse(twist):
    return -1 * twist

def rt_mat(rotation, translation):
    """Combine a rotation/translation into a single homogenous matrix."""
    res = np.identity(4)
    res[:3, :3] = rotation
    res[:3,3] = translation
    return res

def pts_norm(pts):
    return np.sum(norm(pts, axis=0))/pts.shape[1]

def pts_max_len(pts):
    return np.max(norm(pts, axis=0))


# deriv  \frac{\partial R(twist)v}{\partial twist}
#    => -cross_mat(v)
#
# R(twist + delta)v \approx R(twist)v + (-cross_mat(v)) delta

def hproj(pts):
    """Homogenous coordinate projection."""
    return pts[:-1,:] / pts[-1,:]

def dhprojT(pt):
    """Derivative of the homogenous projection.."""
    x, y, z = pt
    return np.array([[1/z, 0],
                     [0, 1/z],
                     [-x/z**2, -y/z**2]])

def dhproj(pt):
    """Derivative of the homogenous projection.."""
    x, y, z = pt
    return np.array([[1/z, 0, -x/z**2],
                     [0, 1/z, -y/z**2]])

def augment(pts):
    """Augment vectors to make them homogenous."""
    return np.vstack((pts, np.ones((1, pts.shape[1]))))

def daugment(pt):
    return pt[:-1,:]


def drt_mat(pt):
    """Change in 3D point position given a change in R and T.  R is axis angle."""
    res = np.zeros((4, 6))
    res[:3, :3] = -cross_mat(pt)
    res[:3, 3:] = np.identity(3)
    res[3,5] = 0   # this was 1,  this seems right
    return res

def dr_mat(pt):
    """Change in 3D point position given a change in R.  R is axis angle."""
    res = np.zeros((4, 3))
    res[:3, :3] = -cross_mat(pt)
    return res

def project_pts(pts, rt_mat, camera, noise=0.0):
    """Project points from 3d to image coordinates."""
    p1 = augment(pts)
    p2 = np.dot(rt_mat, p1)
    p3 = np.dot(camera, p2)
    p4 = hproj(p3)
    if noise > 0.0:
        std = np.std(p4)
        roslog.logdebug("Adding %f noise, %f std.", noise * std, std)
        p5 = p4 + std * noise * np.random.randn(*p4.shape)
    else:
        p5 = p4
    return pts, p1, p2, p3, p4, p5


# Reconstruction code begins
def linear_triangulate_pts(pt0, rt0, cam0, pt1, rt1, cam1):
    """Given two cameras and observed 2D pts, compute 3D pts.
    From Hartley and Zisserman pg 312."""
    P0 = np.dot(cam0, rt0)
    P1 = np.dot(cam1, rt1)
    # note cross(pt0, dot(P0, X)) = 0
    # Assuming that z's are ONE
    x0, y0 = pt0
    x1, y1 = pt1
    A = np.vstack((x0 * P0[2,:] - P0[0,:],
                   y0 * P0[2,:] - P0[1,:],
                   x1 * P1[2,:] - P1[0,:],
                   y1 * P1[2,:] - P1[1,:]))
    U,D,V = np.linalg.svd(A)
    return V[3,:] / V[3,3]


# Creates a jacobian matrix by loading entries for various types of parameters.
def load_pts_jacobian(jacobian, residual_offset, param_offset, pipeline, rt, camera):
    """Load the section of the jacobian related to pt location.
    EG the derivative of the image coordinates given a change in the 3D coordinate of a PT."""
    p3 = pipeline[3]
    num_pts = p3.shape[1]
    poff = param_offset
    roff = residual_offset
    for i in range(num_pts):
        dp5_dp3 = dhproj(p3[:,i])
        dp3_dp1 = np.dot(camera, rt)
        dp3_dp0 = dp3_dp1[:,:-1]
        j = np.dot(dp5_dp3, dp3_dp0)
        r, p = j.shape
        jacobian[roff:roff+r, poff:poff+p] += j
        roff += r
        poff += p
    return roff, poff

def load_3D_pts_jacobian(jacobian, residual_offset, param_offset, pipeline, rt, camera):
    """Load the section of the jacobian related to pt location.
    Assumes direct 3D observation of PTS (in the camera coordinate frame).  So residuals are 3D."""
    p3 = pipeline[3]
    num_pts = p3.shape[1]
    poff = param_offset
    roff = residual_offset
    for i in range(num_pts):
        dp3_dp1 = np.dot(camera, rt)
        dp3_dp0 = dp3_dp1[:,:-1]
        j = dp3_dp0
        r, p = j.shape
        jacobian[roff:roff+r, poff:poff+p] += j
        roff += r
        poff += p
    return roff, poff


def load_camera_RT_jacobian(jacobian, residual_offset, param_offset, pipeline, camera):
    """Load the section of the jacobian related to camera pose: ROTATION and TRANSLATION.
    EG the derivative of the image coordinates given a change in camera pose.
    ASSUMES that rotation is twist. """
    pts, p1, p2, p3, p4, p5 = pipeline
    num_pts = p3.shape[1]
    poff = param_offset
    roff = residual_offset
    for i in range(num_pts):
        dp2_dp0 = drt_mat(pts[:,i])
        dp5_dp3 = dhproj(p3[:,i])
        dp3_dp0 = np.dot(camera, dp2_dp0)
        j = np.dot(dp5_dp3, dp3_dp0)
        r, p = j.shape
        jacobian[roff:roff+r, poff:poff+p] += j
        roff += r
    return roff, poff + p

def load_3D_camera_RT_jacobian(jacobian, residual_offset, param_offset, pipeline, camera):
    """Load the section of the jacobian related to camera pose: ROTATION and TRANSLATION.
    EG the derivative of the observed 3D pt coordinates given a change in camera pose.
    ASSUMES direct 3D observation of PTS (in the camera coordinate frame).  So residuals are 3D.
    ASSUMES that rotation is twist. """
    pts, p1, p2, p3, p4, p5 = pipeline
    num_pts = p3.shape[1]
    poff = param_offset
    roff = residual_offset
    for i in range(num_pts):
        dp2_dp0 = drt_mat(pts[:,i])
        dp3_dp0 = np.dot(camera, dp2_dp0)
        j = dp3_dp0
        r, p = j.shape
        jacobian[roff:roff+r, poff:poff+p] += j
        roff += r
    return roff, poff + p


def load_camera_R_jacobian(jacobian, residual_offset, param_offset, pipeline, camera):
    """Load the section of the jacobian related to camera pose: ROTATION.
    EG the derivative of the image coordinates given a change in camera pose.
    ASSUMES that rotation is twist. """
    pts, p1, p2, p3, p4, p5 = pipeline
    num_pts = p3.shape[1]
    poff = param_offset
    roff = residual_offset
    for i in range(num_pts):
        dp2_dp0 = dr_mat(pts[:,i])
        dp5_dp3 = dhproj(p3[:,i])
        dp3_dp0 = np.dot(camera, dp2_dp0)
        j = np.dot(dp5_dp3, dp3_dp0)
        r, p = j.shape
        jacobian[roff:roff+r, poff:poff+p] += j
        roff += r
    return roff, poff + p


def triangulate_pts(num_iterations, pts3d, rt0, cam0, rt1, cam1, obs0, obs1, alpha=0.0001):
    # DEPRECATED,  sort of.
    """Given two known cameras and the image coordinates for a set of 3d points, find the
    3d coordinates that project to these image points.
    Uses RMS preprojection erorr and relaxation"""
    num_pts = pts3d.shape[1]
    error_last = 10000.0
    for i in range(num_iterations):
        # Pipeline of transformations from 3D to image
        p00, p01, p02, p03, p04, p05 = project_pts(pts3d, rt0, cam0, 0.00)
        p10, p11, p12, p13, p14, p15 = project_pts(pts3d, rt1, cam1, 0.00)
        # p05 and p15 are the predicted projected points

        r0 = obs0 - p05
        r1 = obs1 - p15
        error = (norm(r0)+norm(r1))/num_pts
        if abs(error - error_last) < 0.000001:
            break
        error_last = error
        roslog.logdebug("Reprojection error %03d %f", i, error)
        for i in range(num_pts):
            jT0 = daugment(np.dot(np.dot(rt0.T, cam0.T), dhprojT(p03[:,i])))
            jT1 = daugment(np.dot(np.dot(rt1.T, cam1.T), dhprojT(p13[:,i])))
            jtr = np.dot(jT0, r0[:,i])
            jtr += np.dot(jT1, r1[:,i])
            jtj = np.dot(jT0, jT0.T)
            jtj += np.dot(jT1, jT1.T)
            jtj_inverse = np.linalg.inv(jtj + alpha * np.diag(np.diag(jtj)))
            delta = np.dot(jtj_inverse, jtr)
            pts3d[:,i] += delta

################################################################
# Visualization code
def plot_points(obs, predicted, fignum=1):
    plt.figure(fignum)
    plt.clf()
    plt.scatter(obs[0], obs[1], c='g')
    plt.scatter(predicted[0], predicted[1], c='g')

def scatter_pts(pts, pt_size = 0.01, color = [1, 0, 0, 1]):
    num_pts = pts.shape[0]
    size = pt_size * np.ones((num_pts))
    color = np.ones((num_pts, 1)) * color
    return gl.GLScatterPlotItem(pos=pts, size=size, color=color, pxMode=False)

def plot_3d(pts0, pts1, pt_size=1.0):
    print "Plot 3d"
    w = gl.GLViewWidget()
    w.opts['distance'] = 100
    w.show()
    w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')

    g = gl.GLGridItem(size=QtGui.QVector3D(100, 100, 1))
    w.addItem(g)

    sp = scatter_pts(pts0.T, pt_size, (1.0, 0.0, 0.0, 1.0))
    w.addItem(sp)

    sp = scatter_pts(pts1.T, pt_size, (0.0, 0.0, 1.0, 1.0))
    w.addItem(sp)


################################################################
# Testing/example code below


def test_cross_mat(count=20):
    id = np.identity(3)
    for i in range(count):
        v = np.random.randn(3)
        u = np.random.randn(3)
        c1 = np.cross(u, v)
        c2 = np.dot(cross_mat(u), v)
        if norm(c1-c2) > 0.0001:
            return False
        c11 = np.cross(u, np.cross(u, v))
        c22 = np.dot(np.dot(cross_mat(u), cross_mat(u)), v)
        if norm(c11-c22) > 0.0001:
            return False

        v_mat = twist_mat(v)
        vi_mat = twist_mat(twist_inverse(v))
        pp = np.dot(v_mat, vi_mat)
        if norm(pp - id) > 0.0001:
            return False
    return True

def test_linear_triangulate_pts():
    num_pts = 10
    q0 = axis_angle_q(np.random.randn(3), 0.0)
    t0 = [0, 0, 0]
    q01 = axis_angle_q(np.random.randn(3), 0.1)  # No rotation
    t01 = np.array([0, 10, 0])                   # 10 units in Y
    tp = 3 * np.random.randn(3, num_pts) + np.array([[0, 0, 20]]).T
    rt0 = rt_mat(q_mat(q0), t0)
    rt1 = rt_mat(q_mat(q01), t01)
    camera = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
    # project the true points into the cameras:
    tp00, tp01, tp02, tp03, tp04, tp05 = project_pts(tp, rt0, camera, 0.001)
    tp10, tp11, tp12, tp13, tp14, tp15 = project_pts(tp, rt1, camera, 0.001)

    for i in range(num_pts):
        pt0 = tp05[:,i]
        pt1 = tp15[:,i]
        pp = linear_triangulate_pts(pt0, rt0, camera, pt1, rt1, camera)
        error = np.max(np.abs(tp01[:,i] - pp))
        if error > 0.1:
            print i, tp01[:,i], pp

def test_load_pts_jacobian():
    """Estimate the location of pts in 3D using a single large jacobian approach."""
    num_pts = 15
    q0 = axis_angle_q(np.random.randn(3), 0.0)
    t0 = np.array([0, 0, 0])
    q1 = axis_angle_q(np.random.randn(3), 0.0)  # No rotation
    t1 = np.array([0, 10, 0])                   # 10 units in Y
    true_pts = np.array([[10, 10, 3]]).T * np.random.randn(3, num_pts) + np.array([[0, 0, 20]]).T
    rt0 = rt_mat(q_mat(q0), t0)
    rt1 = rt_mat(q_mat(q1), t1)
    camera = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
    # project the true points into the cameras:
    pipeline0 = project_pts(true_pts, rt0, camera, 0.0)
    pipeline1 = project_pts(true_pts, rt1, camera, 0.0)
    tobs0 = pipeline0[5]
    tobs1 = pipeline1[5]
    pts = np.zeros((3, num_pts))
    for i in range(num_pts):
        pp = linear_triangulate_pts(tobs0[:,i], rt0, camera, tobs1[:,i], rt1, camera)
        pts[:,i] = pp[:3]
    roslog.logdebug("Linear True error         %f", pts_norm(true_pts - pts))

    # Add some noise to ensure that there is a problem to solve.
    pts = pts + 1.0 * np.random.randn(*pts.shape)

    nresiduals = 2 * num_pts * 2 # num dims * num pts * num cameras
    nparams = 3 * num_pts # num dims * num pts
    error_last = 10000.0
    for i in range(10):
        # reproject these new points
        pipeline0 = project_pts(pts, rt0, camera, 0.0)
        pipeline1 = project_pts(pts, rt1, camera, 0.0)

        residuals = np.hstack((tobs0 - pipeline0[5], tobs1 - pipeline1[5]))
        residuals = residuals.reshape(-1, 1, order='F')
        error = pts_norm(residuals)
        roslog.logdebug("RMS residual error            %f", error)
        if abs(error - error_last) < 0.0001:
            break
        error_last = error

        jacobian = np.zeros((nresiduals, nparams))
        load_pts_jacobian(jacobian, 0, 0, pipeline0, rt0, camera)
        load_pts_jacobian(jacobian, num_pts*2, 0, pipeline1, rt1, camera)

        jT = jacobian.T
        delta_pts = np.dot(np.linalg.inv(np.dot(jT, jacobian)), np.dot(jT, residuals))
        delta_pts = delta_pts.reshape(3, -1, order='F')
        pts = pts + delta_pts
        roslog.logdebug("RMS True error            %f", pts_norm(true_pts - pts))

def test_load_camera_jacobian():
    """Estimate the pose of a camera from the location of 3D pts projected into camera view."""
    num_pts = 100
    q = q_identity()
    t = np.array([0, 0, 0])
    rt = rt_mat(q_mat(q), t)
    camera = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
    pts = np.array([[10, 10, 3]]).T * np.random.randn(3, num_pts) + np.array([[0, 0, 20]]).T
    pipeline = project_pts(pts, rt, camera, 0.001)
    tobs = pipeline[5]

    qguess = axis_angle_q(np.random.randn(3), 0.1)
    tguess = t + np.random.randn(3)

    qnew = qguess.copy()
    tnew = tguess.copy()

    nresiduals = 2 * num_pts # num dims * num pts
    nparams = 6 # axis angle + translate
    error_last = 10000.0
    for i in range(10):
        rtnew = rt_mat(q_mat(qnew), tnew)
        pipeline = project_pts(pts, rtnew, camera, 0.0)

        residuals = tobs - pipeline[5]
        residuals = residuals.reshape(-1, 1, order='F')

        jacobian = np.zeros((nresiduals, nparams))
        load_camera_RT_jacobian(jacobian, 0, 0, pipeline, camera)

        jT = jacobian.T
        jTj = np.dot(jT, jacobian)
        delta = np.dot(np.linalg.inv(jTj), np.dot(jT, residuals))
        delta = delta.reshape(-1)
        qdelta = twist_q(delta[:3]) # using twist for derivatives
        qnew = q_compose(qdelta, qnew)
        tnew = tnew + delta[3:]

        error = pts_norm(residuals)
        roslog.logdebug("RMS residuals            %f", error)
        if abs(error - error_last) < 0.000001:
            break
        error_last = error

    roslog.logdebug("TRUE     %s %s: q, t", q, t)
    roslog.logdebug("GUESS    %s %s: q, t", qguess, tguess)
    roslog.logdebug("ESTIMATE %s %s: q, t", qnew, tnew)

def test_stereo_bundle_adjust():
    """Estimate the location of pts in 3D using a single large jacobian approach."""
    num_pts = 50
    # Generate the two cameras 0 and 1
    q0 = axis_angle_q(np.random.randn(3), 0.0)
    t0 = np.array([0, 0, 0])
    rt0 = rt_mat(q_mat(q0), t0)

    q1 = axis_angle_q(np.random.randn(3), 0.0)     # random orientation
    t1 = 0.0 * np.random.randn(3) + np.array([0, 10, 0]) # baseline + random position
    baseline_length = norm(t1)
    rt1 = rt_mat(q_mat(q1), t1)

    # Boring camera matrix
    camera = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])

    # Generate random 3D points
    true_pts = np.array([[10, 10, 3]]).T * np.random.randn(3, num_pts) + np.array([[0, 0, 20]]).T

    # project the true points into the cameras (these are the observations)
    pipeline0 = project_pts(true_pts, rt0, camera, 0.0)
    pipeline1 = project_pts(true_pts, rt1, camera, 0.0)
    tobs0 = pipeline0[5]
    tobs1 = pipeline1[5]

    # Perturb the second camera (first is assumed fixed and correct)
    angle_error = 0.1
    axis_error = np.random.randn(3)
    q1_wrong = q_compose(axis_angle_q(axis_error, angle_error), q0)
    t1_wrong = t1 + 1.0 * np.random.randn(3)
    q1_est = q1_wrong.copy()
    t1_est = t1_wrong.copy()

    rt1_wrong = rt_mat(q_mat(q1_wrong), t1_wrong)

    pts = np.zeros((3, num_pts))
    for i in range(num_pts):
        pp = linear_triangulate_pts(tobs0[:,i], rt0, camera, tobs1[:,i], rt1_wrong, camera)
        pts[:,i] = pp[:3]
    roslog.logdebug("Linear True error         %f", pts_norm(true_pts - pts))

    pipeline0 = project_pts(pts, rt0, camera, 0.0)
    pipeline1 = project_pts(pts, rt1_wrong, camera, 0.0)

    residuals = np.hstack((tobs0 - pipeline0[5], tobs1 - pipeline1[5]))
    residuals = residuals.reshape(-1, 1, order='F')
    error = pts_norm(residuals)
    roslog.logdebug("RMS residual error            %f", error)

    nresiduals = 2 * num_pts * 2 # num dims * num pts * num cameras
    nparams = 1 * 6 + 3 * num_pts # cameras + points
    error_last = 10000.0
    for i in range(30):
        rt1_est = rt_mat(q_mat(q1_est), t1_est)
        # reproject these new points
        pipeline0 = project_pts(pts, rt0,     camera, 0.0)
        pipeline1 = project_pts(pts, rt1_est, camera, 0.0)

        residuals = np.hstack((tobs0 - pipeline0[5], tobs1 - pipeline1[5]))
        residuals = residuals.reshape(-1, 1, order='F')
        error = pts_norm(residuals)
        roslog.logdebug("RMS residual error            %f", error)
        if abs(error - error_last) < 0.00001:
            break
        error_last = error

        jacobian = np.zeros((nresiduals, nparams))

        roff, poff = load_camera_RT_jacobian(jacobian, num_pts*2, 0, pipeline1, camera)
        # Points are effected by both cameras
        load_pts_jacobian(jacobian, 0,         poff, pipeline0, rt0,     camera)
        load_pts_jacobian(jacobian, num_pts*2, poff, pipeline1, rt1_est, camera)

        jT = jacobian.T
        jTj = np.dot(jT, jacobian)
        alpha = 0.9999
        delta = np.dot(np.linalg.inv(alpha * jTj + (1.0 - alpha) * np.diag(np.diag(jTj))), np.dot(jT, residuals))

        # extract the parameters
        delta = delta.reshape(-1, order='F')
        delta_q1 = delta[:3]
        delta_t1 = delta[3:6]
        delta_pts = delta[6:].reshape(3, -1, order='F')

        # update the pose of the second camera
        q1_est = q_compose(twist_q(delta_q1), q1_est)
        t1_est = t1_est + delta_t1
        pts = pts + delta_pts
        roslog.logdebug("RMS True error            %f", pts_norm(true_pts - pts))

    plot_points(tobs0, pipeline0[5], 1)
    plot_points(tobs1, pipeline1[5], 2)
    roslog.logdebug("INITIAL  %s %s: q, t", q1_wrong, t1_wrong)
    roslog.logdebug("ESTIMATE %s %s: q, t", q1_est, t1_est)
    roslog.logdebug("ESTIMATE %s %s: q, t", q1_est, norm(t1) * t1_est/norm(t1_est))
    roslog.logdebug("TRUE     %s %s: q, t", q1, t1)
    return true_pts, pts

def test_stereo_bundle_adjust_cam1_rotation():
    """
    Stereo: Estimate the rotation of cam1, assuming cam0 and the
    position of cam1 are known.  Uses BA to estimate the 3D locations
    of 3D pts as a byproduct.
    """
    num_pts = 50
    # Generate the two cameras 0 and 1
    # Camera 0 is at the origin looking down the z_axis.
    deformation_angle = 0.01
    q0 = q_identity()
    t0 = np.array([0, 0, 0])
    rt0 = rt_mat(q_mat(q0), t0)

    # Cam1 is at a rotated position and orientation
    r0 = q_mat(axis_angle_q(np.random.randn(3), deformation_angle))
    q1 = axis_angle_q(np.random.randn(3), deformation_angle)     # random orientation
    baseline = np.array([0, 10, 0])
    t1 = np.dot(r0, baseline) # true
    rt1 = rt_mat(q_mat(q1), t1)

    # Boring camera matrix
    camera = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])

    # Generate random 3D points
    true_pts = np.array([[10, 10, 3]]).T * np.random.randn(3, num_pts) + np.array([[0, 0, 20]]).T

    # project the true points into the cameras (these are the observations)
    pipeline0 = project_pts(true_pts, rt0, camera, 0.0)
    pipeline1 = project_pts(true_pts, rt1, camera, 0.0)
    tobs0 = pipeline0[5]
    tobs1 = pipeline1[5]

    # Assume that cam0 is the identity, cam1 is estimated in that space
    angle_error = 0.1
    q0_wrong = q_identity() # use the coordinate frame of cam0
    t0_wrong = np.array([0, 0, 0])
    t1_wrong = baseline
    q1_wrong = q_compose(axis_angle_q(np.random.randn(3), angle_error), q0)
    q1_est = q1_wrong.copy()
    t1_est = t1_wrong.copy()

    rt0_wrong = rt_mat(q_mat(q0_wrong), t0_wrong)
    rt1_wrong = rt_mat(q_mat(q1_wrong), t1_wrong)

    pts = np.zeros((3, num_pts))
    for i in range(num_pts):
        pp = linear_triangulate_pts(tobs0[:,i], rt0_wrong, camera, tobs1[:,i], rt1_wrong, camera)
        pts[:,i] = pp[:3]
    roslog.logdebug("Linear True error         %f", pts_norm(true_pts - pts))

    pipeline0 = project_pts(pts, rt0_wrong, camera, 0.0)
    pipeline1 = project_pts(pts, rt1_wrong, camera, 0.0)

    residuals = np.hstack((tobs0 - pipeline0[5], tobs1 - pipeline1[5]))
    residuals = residuals.reshape(-1, 1, order='F')
    error = pts_norm(residuals)
    roslog.logdebug("RMS residual error            %f", error)

    nresiduals = 2 * num_pts * 2 # num dims * num pts * num cameras
    nparams = 1 * 3 + 3 * num_pts # cameras + points
    error_last = 10000.0
    for i in range(30):
        rt1_est = rt_mat(q_mat(q1_est), t1_est)
        # reproject these new points
        pipeline0 = project_pts(pts, rt0_wrong,     camera, 0.0)
        pipeline1 = project_pts(pts, rt1_est, camera, 0.0)

        residuals = np.hstack((tobs0 - pipeline0[5], tobs1 - pipeline1[5]))
        residuals = residuals.reshape(-1, 1, order='F')
        error = pts_norm(residuals)
        roslog.logdebug("RMS residual error            %f", error)
        if abs(error - error_last) < 0.00000001:
            break
        error_last = error

        jacobian = np.zeros((nresiduals, nparams))

        roff, poff = load_camera_R_jacobian(jacobian, num_pts*2, 0, pipeline1, camera)
        # Points are affected by both cameras
        load_pts_jacobian(jacobian, 0,         poff, pipeline0, rt0_wrong,     camera)
        load_pts_jacobian(jacobian, num_pts*2, poff, pipeline1, rt1_est, camera)

        jT = jacobian.T
        jTj = np.dot(jT, jacobian)
        alpha = 0.9999
        delta = np.dot(np.linalg.inv(alpha * jTj + (1.0 - alpha) * np.diag(np.diag(jTj))), np.dot(jT, residuals))

        # extract the parameters
        delta = delta.reshape(-1, order='F')
        off = 0
        delta_q1 = delta[off:off+3]; off += 3
        delta_pts = delta[off:].reshape(3, -1, order='F')

        # update the pose of the second camera
        q1_est = q_compose(twist_q(delta_q1), q1_est)
        pts = pts + delta_pts
        roslog.logdebug("RMS True error            %f", pts_norm(true_pts - pts))

    plot_points(tobs0, pipeline0[5], 1)
    plot_points(tobs1, pipeline1[5], 2)

    roslog.logdebug("INITIAL  %s %s: q, t", q1_wrong, t1_wrong)
    roslog.logdebug("ESTIMATE %s %s: q, t", q1_est, t1_est)
    roslog.logdebug("TRUE     %s %s: q, t", q1, t1)
    roslog.logdebug("Initial camera angles     %s", q_angle(q_compose(q0, q_inverse(q1))) - 2 * math.pi)
    roslog.logdebug("Estimated camera angles   %s", q_angle(q_compose(q0_wrong, q_inverse(q1_est))) - 2 * math.pi)
    return q0, q1, q0_wrong, q1_est



def test_stereo_bundle_adjust_baseline():
    """Estimate the location of pts in 3D using a single large jacobian approach."""
    num_pts = 50
    # Generate the two cameras 0 and 1
    deformation_angle = 0.01
    q0 = axis_angle_q(np.random.randn(3), 0.0)
    t0 = np.array([0, 0, 0])
    rt0 = rt_mat(q_mat(q0), t0)

    q1 = axis_angle_q(np.random.randn(3), deformation_angle)     # random orientation
    t1 = 0.5 * np.random.randn(3) + np.array([0, 10, 0]) # baseline + random position
    rt1 = rt_mat(q_mat(q1), t1)

    baseline_length = norm(t1)
    rt1 = rt_mat(q_mat(q1), t1)

    # Boring camera matrix
    camera = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])

    # Generate random 3D points
    true_pts = np.array([[10, 10, 3]]).T * np.random.randn(3, num_pts) + np.array([[0, 0, 20]]).T

    # project the true points into the cameras (these are the observations)
    pipeline0 = project_pts(true_pts, rt0, camera, 0.005)
    pipeline1 = project_pts(true_pts, rt1, camera, 0.005)
    tobs0 = pipeline0[5]
    tobs1 = pipeline1[5]

    # Perturb the second camera (first is assumed fixed and correct)
    angle_error = 0.1
    axis_error = np.random.randn(3)
    q1_wrong = q_compose(axis_angle_q(axis_error, angle_error), q0_wrong)
    t1_wrong = t1 + 0.1 * np.random.randn(3)
    q1_est = q1_wrong.copy()
    t1_est = t1_wrong.copy()
    roslog.logdebug("EST baseline             %f", norm(t1_est))

    rt1_wrong = rt_mat(q_mat(q1_wrong), t1_wrong)

    pts = np.zeros((3, num_pts))
    for i in range(num_pts):
        pp = linear_triangulate_pts(tobs0[:,i], rt0, camera, tobs1[:,i], rt1_wrong, camera)
        pts[:,i] = pp[:3]
    roslog.logdebug("Linear True error         %f", pts_norm(true_pts - pts))

    pipeline0 = project_pts(pts, rt0, camera, 0.0)
    pipeline1 = project_pts(pts, rt1_wrong, camera, 0.0)

    plot_points(tobs0, pipeline0[5], 3)
    plot_points(tobs1, pipeline1[5], 4)

    residuals = np.hstack((tobs0 - pipeline0[5], tobs1 - pipeline1[5]))
    residuals = residuals.reshape(-1, 1, order='F')
    error = pts_norm(residuals)
    roslog.logdebug("RMS residual error            %f", error)

    nresiduals = 2 * num_pts * 2 + 1 # num dims * num pts * num cameras
    nparams = 1 * 6 + 3 * num_pts # cameras + points
    error_last = 10000.0
    for i in range(30):
        rt1_est = rt_mat(q_mat(q1_est), t1_est)
        # reproject these new points
        pipeline0 = project_pts(pts, rt0,     camera, 0.0)
        pipeline1 = project_pts(pts, rt1_est, camera, 0.0)

        baseline_error = 1.0 * (norm(t1_est) - baseline_length)
        residuals = np.hstack((tobs0 - pipeline0[5], tobs1 - pipeline1[5]))
        residuals = residuals.reshape(-1, 1, order='F')
        residuals = np.vstack((residuals, [-baseline_error]))
        error = pts_norm(residuals)
        roslog.logdebug("RMS residual error            %f", error)
        if abs(error - error_last) < 0.00000001:
            break
        error_last = error

        jacobian = np.zeros((nresiduals, nparams))

        roff, poff = load_camera_RT_jacobian(jacobian, num_pts*2, 0, pipeline1, camera)
        # Points are effected by both cameras
        load_pts_jacobian(jacobian, 0,         poff, pipeline0, rt0,     camera)
        load_pts_jacobian(jacobian, num_pts*2, poff, pipeline1, rt1_est, camera)
        jacobian[-1,3:6] = t1_est / norm(t1_est)

        jT = jacobian.T
        jTj = np.dot(jT, jacobian)
        alpha = 1.0
        delta = np.dot(np.linalg.inv(alpha * jTj + (1.0 - alpha) * np.diag(np.diag(jTj))), np.dot(jT, residuals))

        # extract the parameters
        delta = delta.reshape(-1, order='F')
        delta_q1 = delta[:3]
        delta_t1 = delta[3:6]
        delta_pts = delta[6:].reshape(3, -1, order='F')

        # update the pose of the second camera
        q1_est = q_compose(twist_q(delta_q1), q1_est)
        t1_est = t1_est + delta_t1
        pts = pts + delta_pts
        roslog.logdebug("baseline                 %f", norm(t1_est))
        roslog.logdebug("RMS True error            %f", pts_norm(true_pts - pts))

    plot_points(tobs0, pipeline0[5], 1)
    plot_points(tobs1, pipeline1[5], 2)
    roslog.logdebug("INITIAL  %s %s: q, t", q1_wrong, t1_wrong)
    roslog.logdebug("ESTIMATE %s %s: q, t", q1_est, t1_est)
    roslog.logdebug("ESTIMATE %s %s: q, t", q1_est, norm(t1) * t1_est/norm(t1_est))
    roslog.logdebug("TRUE     %s %s: q, t", q1, t1)
    return true_pts, pts


def generate_cameras(num_cameras, angle, shift, noise):
    q0 = axis_angle_q(np.random.randn(3), 0.0)
    t0 = np.array([0, 0, 0])
    true_cameras = [(q0, t0)]
    next_t = t0
    for i in range(num_cameras-1):
        next_q = axis_angle_q(np.random.randn(3), angle)  # No rotation
        next_t = noise * np.random.randn(3) + next_t + np.array(shift)
        true_cameras.append((next_q, next_t))
    return true_cameras

def randomize_cameras(cameras, angle_error, trans_noise):
    num_cameras = len(cameras)
    q, t = cameras[0]
    new_cameras = [(q, t)]
    for i in range(1, num_cameras):
        q,t = cameras[i]
        axis_error = np.random.randn(3)
        q_wrong = q_compose(axis_angle_q(axis_error, angle_error), q)
        t_wrong = t + trans_noise * np.random.randn(3)
        new_cameras.append((q_wrong, t_wrong))
    return new_cameras

def linear_initialize_pts(cam0, intrinsic0, cam1, intrinsic1, obs0, obs1):
    num_pts = obs0.shape[1]
    q0, t0 = cam0
    rt0 = rt_mat(q_mat(q0), t0)
    q1, t1 = cam1
    rt1 = rt_mat(q_mat(q1), t1)

    pts = np.zeros((3, num_pts))
    for i in range(num_pts):
        pp = linear_triangulate_pts(obs0[:,i], rt0, intrinsic0, obs1[:,i], rt1, intrinsic1)
        pts[:,i] = pp[:3]
    return pts


def test_bundle_adjust():
    """Joint estimate the position of pts and cameras."""
    num_pts = 100
    num_cameras = 50
    # Generate the cameras
    true_cameras = generate_cameras(num_cameras, 0.1, [0.0, 3.0, 0.0], 0.0)

    # Generate random 3D points
    true_pts = np.array([[10, 10, 3]]).T * np.random.randn(3, num_pts) + np.array([[0, 0, 20]]).T
    camera = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
    # project the true points into the cameras (these are the observations)
    true_obs = []
    for q, t in true_cameras:
        rt = rt_mat(q_mat(q), t)
        pipeline = project_pts(true_pts, rt, camera, 0.0)
        true_obs.append(pipeline[5])

    initial_cameras = randomize_cameras(true_cameras, 0.1, 1.0)
    estimated_cameras = copy.copy(initial_cameras)

    # Important to triangulate from poses that are far apart (relative to the noise in the initial estimate of pose)
    pts = linear_initialize_pts(initial_cameras[0], camera, initial_cameras[10], camera, true_obs[0], true_obs[10])
    roslog.logdebug("Linear True error         %f", pts_norm(true_pts - pts))

    camera_model_size = 6
    ncamera_params = camera_model_size * (num_cameras - 1)
    nparams = ncamera_params + 3 * num_pts # cameras + points, first camera is fixed
    nresiduals = 2 * num_pts * num_cameras # num dims * num pts * num cameras
    error_last = 10000.0
    for rounds in range(30):

        jacobian = np.zeros((nresiduals, nparams))
        residuals = None
        for i in range(num_cameras):
            q, t = estimated_cameras[i]
            tobs = true_obs[i]
            rt = rt_mat(q_mat(q), t)
            pipeline = project_pts(pts, rt, camera, 0.0)
            if residuals is None:
                residuals = tobs - pipeline[5]
            else:
                residuals = np.hstack((residuals, tobs - pipeline[5]))
            pt_offset = i * 2 * num_pts # offset to pts visible in this camera
            if i > 0:
                load_camera_RT_jacobian(jacobian, pt_offset, (i-1) * camera_model_size, pipeline, camera)
            load_pts_jacobian(jacobian, pt_offset, ncamera_params, pipeline, rt, camera)

        residuals = residuals.reshape(-1, 1, order='F')
        error = pts_norm(residuals)
        roslog.logdebug("RMS residual error            %f", error)
        if abs(error - error_last) < 0.00001:
            break
        error_last = error

        jT = jacobian.T
        delta = np.dot(np.linalg.inv(np.dot(jT, jacobian) + 0.00001 * np.identity(nparams)), np.dot(jT, residuals))
        delta = delta.reshape(-1, order='F')

        for i in range(1, num_cameras):
            q, t = estimated_cameras[i]
            param_offset = (i-1) * camera_model_size
            delta_q = delta[param_offset:param_offset+3]
            delta_t = delta[param_offset+3:param_offset+6]
            q_new = q_compose(twist_q(delta_q), q)
            t_new = delta_t + t
            estimated_cameras[i] = (q_new, t_new)

        delta_pts = delta[ncamera_params:].reshape(3, -1, order='F')
        pts = pts + delta_pts
        roslog.logdebug("RMS True error            %f", pts_norm(true_pts - pts))
    for i in range(num_cameras):
        q, t = true_cameras[i]
        q0, t0 = initial_cameras[i]
        qe, te = estimated_cameras[i]
        roslog.logdebug("ESTIMATE %s %s %s: q, t, angle", qe, te, small_angle(q_angle(q_compose(qe, q_inverse(qe)))))


def test_3D_marker_bundle_adjust():
    """Joint estimate the position of pts and cameras."""
    num_pts = 8
    num_cameras = 300
    # Generate the cameras
    true_cameras = generate_cameras(num_cameras, 0.1, [0.0, 1.0, 0.0], 0.0)

    # Generate random 3D points
    true_pts = np.array([[10, 10, 3]]).T * np.random.randn(3, num_pts) + np.array([[0, 0, 20]]).T
    camera = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
    # project the true points into the cameras (these are the observations)
    true_obs = []
    for q, t in true_cameras:
        rt = rt_mat(q_mat(q), t)
        pipeline = project_pts(true_pts, rt, camera, 0.0)
        true_obs.append(pipeline[3] + 0.01 * np.random.randn(*pipeline[3].shape))

    initial_cameras = randomize_cameras(true_cameras, 0.5, 1.0)
    estimated_cameras = copy.copy(initial_cameras)

    pts = true_pts + 10.0 * np.random.randn(*true_pts.shape)
    roslog.logdebug("Linear True error         %f", pts_norm(true_pts - pts))

    camera_model_size = 6
    ncamera_params = camera_model_size * (num_cameras - 1)
    nparams = ncamera_params + 3 * num_pts # cameras + points, first camera is fixed
    nresiduals = 3 * num_pts * num_cameras # num dims * num pts * num cameras
    error_last = 10000.0
    for rounds in range(30):

        jacobian = np.zeros((nresiduals, nparams))
        residuals = None
        for i in range(num_cameras):
            q, t = estimated_cameras[i]
            tobs = true_obs[i]
            rt = rt_mat(q_mat(q), t)
            pipeline = project_pts(pts, rt, camera, 0.0)
            if residuals is None:
                residuals = tobs - pipeline[3]
            else:
                residuals = np.hstack((residuals, tobs - pipeline[3]))
            pt_offset = i * 3 * num_pts # offset to pts visible in this camera
            if i > 0:
                load_3D_camera_RT_jacobian(jacobian, pt_offset, (i-1) * camera_model_size, pipeline, camera)
            load_3D_pts_jacobian(jacobian, pt_offset, ncamera_params, pipeline, rt, camera)

        residuals = residuals.reshape(-1, 1, order='F')
        error = pts_norm(residuals)
        roslog.logdebug("RMS residual error            %f", error)
        if abs(error - error_last) < 0.00001:
            break
        error_last = error

        jT = jacobian.T
        delta = np.dot(np.linalg.inv(np.dot(jT, jacobian) + 0.000001 * np.identity(nparams)), np.dot(jT, residuals))
        delta = delta.reshape(-1, order='F')

        for i in range(1, num_cameras):
            q, t = estimated_cameras[i]
            param_offset = (i-1) * camera_model_size
            delta_q = delta[param_offset:param_offset+3]
            delta_t = delta[param_offset+3:param_offset+6]
            q_new = q_compose(twist_q(delta_q), q)
            t_new = delta_t + t
            estimated_cameras[i] = (q_new, t_new)
        delta_pts = delta[ncamera_params:].reshape(3, -1, order='F')
        pts = pts + delta_pts
        roslog.logdebug("RMS True error            %f", pts_norm(true_pts - pts))

    for i in range(num_cameras):
        q, t = true_cameras[i]
        q0, t0 = initial_cameras[i]
        qe, te = estimated_cameras[i]
        roslog.logdebug("ERROR %s %s", small_angle(q_angle(q_compose(q, q_inverse(qe)))), norm(t-te))
        # roslog.logdebug("TRUE     %s %s", q, t)
        # roslog.logdebug("GUESS    %s %s", q0, t0)
        # roslog.logdebug("ESTIMATE %s %s %s: q, t, angle", qe, te, small_angle(q_angle(q_compose(qe, q_inverse(qe)))))


