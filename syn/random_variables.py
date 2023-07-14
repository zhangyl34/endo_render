import numpy as np
import scipy.stats as sstats
from autolab_core import CameraIntrinsics, RigidTransform, transformations
from autolab_core.utils import sph2cart


class CameraRandomVariable(object):
    """Uniform distribution over camera poses and intrinsics about a viewsphere over a planar worksurface.
    The camera is positioned pointing towards (0,0,0).
    """

    def __init__(self, config):
        # read params
        self.config = config
        self._parse_config(config)


    def _parse_config(self, config):
        """Reads parameters from the config into class members."""
        # camera params
        self.frame = config["name"]                 # camera
        self.focal_length = config["focal_length"]
        self.im_height = config["im_height"]
        self.im_width = config["im_width"]
        self.mean_cx = float(self.im_width) / 2.0
        self.mean_cy = float(self.im_height) / 2.0

    def camera_to_world_pose(self):
        """Convert spherical coords to a camera pose in the world."""
        # generate camera center from spherical coords
        camera_center = np.array([0, 0, 0])

        # find the canonical camera x and y axes
        camera_x = np.array([-1, 0, 0])
        camera_y = np.array([0, -1, 0])
        camera_z = np.array([0, 0, 1])
        # get w^T_cam
        R = np.vstack((camera_x, camera_y, camera_z)).T
        T_camera_world = RigidTransform(
            R, camera_center, from_frame=self.frame, to_frame="world"
        )

        return T_camera_world

    def sample(self, size=1):
        """Sample random variables from the model.
        Parameters
        ----------
        cart_pos: [x, y, 0]
            position of the cart.
        size : int
            number of sample to take
        Returns
        -------
        :obj:`list` of :obj:`CameraSample`
            sampled camera intrinsics and poses
        """
        samples = []
        for i in range(size):
            # sample camera params
            focal = self.focal_length 
            cx = self.mean_cx   
            cy = self.mean_cy    

            # convert to pose and intrinsics
            pose = self.camera_to_world_pose()
            intrinsics = CameraIntrinsics(
                self.frame,
                fx=focal,
                fy=focal,
                cx=cx,
                cy=cy,
                skew=0.0,
                height=self.im_height,
                width=self.im_width,
            )

            # convert to camera pose
            samples.append((pose, intrinsics))

        # not a list if only 1 sample
        if size == 1:
            return samples[0]
        return samples
