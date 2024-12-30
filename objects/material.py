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

    def get_color(self, ray: Ray, normal: np.ndarray, point: np.ndarray, scene: Scene, self_obj):
        ambient_light = self.cfg.ambient.copy()
        diffuse_light = np.array([0.0, 0.0, 0.0])
        specular_light = np.array([0.0, 0.0, 0.0])
        emittance = self.cfg.emittance.copy()
        for l in scene.lights:
            if l is self_obj:
                continue
            l_dir = l.geometry.get_light_source_dir(point)
            light_ray_cfg = RayConfig()
            light_ray_cfg.origin = point.copy()
            light_ray_cfg.dir = l_dir
            light_ray = Ray(light_ray_cfg)
            light_intersection = l.geometry.intersect_with_ray(light_ray)
            light_intersection_dist = np.linalg.norm(light_intersection - point)
            assert light_intersection is not None
            is_shadow = False
            for obj in scene.objects:  # TODO: intersection loop?
                if obj is not l and obj is not self_obj:
                    intersection = obj.geometry.intersect_with_ray(light_ray)
                    if intersection is not None and np.linalg.norm(intersection - point) < light_intersection_dist:
                        is_shadow = True
                        break
            if is_shadow:
                continue
            diffuse_light = l_dir.dot(normal) * self.cfg.diffuse
            r_dir = Ray.reflect_dir(l_dir, normal)  # needs the direction from the light to the point
            specular_light = (r_dir.dot(ray.cfg.dir)) ** self.cfg.glossiness * self.cfg.specular
        return ambient_light + specular_light + diffuse_light + emittance
