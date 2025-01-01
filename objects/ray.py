from objects.config import Config
import numpy as np
from objects.transform import affine_transform


class RayConfig(Config):
    def __init__(self):
        super().__init__()
        self.dir = np.array([0, 0, -1])
        self.origin = np.array([0, 0, 0])

    def validate(self):
        assert self.dir.shape[0] == 3
        assert self.origin.shape[0] == 3


class Ray:
    def __init__(self, cfg: RayConfig):
        self.cfg = cfg
        self.cfg.dir /= np.linalg.norm(self.cfg.dir)

    def apply_transform(self, transform_matrix):
        o = self.cfg.origin
        d = self.cfg.dir
        unit_point = o + d
        o_t = affine_transform(o, transform_matrix)
        unit_point_t = affine_transform(unit_point, transform_matrix)
        d_t = unit_point_t - o_t
        d_t /= np.linalg.norm(d_t)
        t_ray_cfg = RayConfig()
        t_ray_cfg.origin = o_t
        t_ray_cfg.dir = d_t
        return Ray(t_ray_cfg)

    @staticmethod
    def sample_ray_from_hemisphere():
        pass

    @staticmethod
    def reflect_ray_dir(ray_dir: np.ndarray, normal: np.ndarray):
        # TODO: remove if takes time
        assert np.isclose(np.linalg.norm(ray_dir), 1)
        assert np.isclose(np.linalg.norm(normal), 1)
        return ray_dir - 2 * ray_dir.dot(normal) * normal
