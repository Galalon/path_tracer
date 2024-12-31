from objects.config import Config
import numpy as np
from objects.ray import Ray, RayConfig
from objects.scene import Scene


class MaterialConfig(Config):
    def __init__(self):
        super().__init__()
        self.emittance = np.array([0.0, 0.0, 0.0])
        self.diffuse = np.array([1.0, 1.0, 1.0])  # solid color for now

    def validate(self):
        return


class PhongMaterialConfig(MaterialConfig):
    def __init__(self):
        super().__init__()
        self.specular = np.array([0.0, 0.0, 0.0])
        self.glossiness = 5
        self.ambient = np.array([0.0, 0.0, 0.0])
        self.k_diffuse = 1
        self.k_specular = 1
        self.k_ambient = 1


class PhongMaterial:
    def __init__(self, cfg: PhongMaterialConfig):
        self.cfg = cfg

    def get_color_per_light(self, view_ray: Ray, light_ray: Ray, point: np.ndarray, normal: np.ndarray):
        if light_ray is None:
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
        l_dir = light_ray.cfg.dir
        diffuse_light = l_dir.dot(normal) * self.cfg.diffuse
        r_dir = Ray.reflect_dir(-l_dir, normal)  # refloct works for dirs from the light to a point
        specular_light = (r_dir.dot(
            -view_ray.cfg.dir)) ** self.cfg.glossiness * self.cfg.specular  # specular = r*v where v - direction towards camera
        diffuse_light = np.maximum(diffuse_light, 0)
        specular_light = np.maximum(specular_light, 0)
        return diffuse_light, specular_light
