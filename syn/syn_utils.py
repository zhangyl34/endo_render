import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def in_hull(p, hull):
    ''' input:
    p: (N,3)
    output:
    (1,N) bool'''
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0
    
def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


# copy from mask.py
def quat2R(quat):
    assert len(quat) == 4
    mat = R.from_quat(quat).as_matrix()
    return mat

def euler2quat(rot):
    assert len(rot) == 3
    quat = R.from_euler('xyz', rot, degrees=True).as_quat()
    return quat


def cal_z_angular_err(quat_1, quat_2):
    R1 = quat2R(quat_1)
    R2 = quat2R(quat_2)
    z1 = np.matmul([0, 0, 1], R1)
    z2 = np.matmul([0, 0, 1], R2)
    angle = np.arccos(np.dot(z1, z2))
    return np.degrees(angle)

def cal_x_angular_err(R1, quat_2):
    R2 = quat2R(quat_2)
    z1 = np.matmul([1, 0, 0], R1)
    z2 = np.matmul([0, 0, 1], R2)
    angle = np.arccos(np.dot(z1, z2))
    return np.degrees(angle)

def data_filter_rotation(rot):
    rot_ori = [0, 0, 0]
    angle = cal_z_angular_err(euler2quat(rot), euler2quat(rot_ori))
    if angle <= 45 or angle >= 100:
        return False
    else:
        return True
    
def data_filter_stem_rotation(R):
    rot_ori = [0, 0, 0]
    angle = cal_x_angular_err(R, euler2quat(rot_ori))
    if angle <= 80 or angle >= 170:
        # print(angle)
        return False
    else:
        # print(angle)
        return True
