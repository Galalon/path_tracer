from objects.config import Config
import numpy as np


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

    @staticmethod
    def sample_ray_from_hemisphere():
        pass
    @staticmethod
    def reflect_dir(ray_dir:np.ndarray,normal:np.ndarray):
        #TODO: remove if takes time
        assert np.isclose(np.linalg.norm(ray_dir),1)
        assert np.isclose(np.linalg.norm(normal), 1)
        return ray_dir - 2 * ray_dir.dot(normal) * normal



