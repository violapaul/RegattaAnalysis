
import numpy as np
from numpy.linalg import norm
import math

EPSILON = np.finfo(float).eps

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
    if math.fabs(angle) < EPSILON:
        return q_identity()
    else:
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
        print(f"Adding {noise*std} noise, {std} std.")
        p5 = p4 + std * noise * np.random.randn(*p4.shape)
    else:
        p5 = p4
    return pts, p1, p2, p3, p4, p5



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
