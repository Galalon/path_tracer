from abc import ABC
from objects.config import Config
import numpy as np


# Scene class to encapsulate parameters

class SceneConfig(Config):
    def __init__(self):
        super().__init__()
        self.buffer_size_hw = (None, None)


class Scene(ABC):
    def __init__(self, cfg: SceneConfig):
        self.cfg = cfg


class MandelbrotConfig(SceneConfig):
    def __init__(self):
        super().__init__()
        self.bbox = (-2.0, 1.0, -1.5, 1.5)  # (xmin, xmax, ymin, ymax)
        self.max_iters = 100  # Maximum iterations for Mandelbrot
        self.radius = 2.0  # Radius of convergence
        self.resolution_x = 800

    def validate(self):
        """Custom validation for the MandelbrotConfig."""
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive.")
        if not isinstance(self.bbox, tuple) or len(self.bbox) != 4:
            raise ValueError("bbox must be a tuple with 4 elements.")


# complex plane for mandelbrot example
class MandelbrotScene(Scene):
    def __init__(self, cfg: MandelbrotConfig):
        super().__init__(cfg)
        self.cfg = cfg  # technically unnecessary but it fucks up the UI
        xmin, xmax, ymin, ymax = self.cfg.bbox
        self.cfg.resolution_y = self.cfg.resolution_x * (ymax - ymin) / (xmax - xmin)  # TODO: either one
        self.cfg.resolution_y = int(self.cfg.resolution_y)
        self.cfg.buffer_size_hw = (self.cfg.resolution_y, self.cfg.resolution_x)


from objects.camera import CameraConfig, Camera
from objects.render_object import RenderObjectConfig, RenderObject


class RenderSceneConfig(SceneConfig):
    def __init__(self):
        super().__init__()
        self.camera_cfg = CameraConfig()
        self.objects_cfg = [RenderObjectConfig()]
        self.buffer_size_hw = (480, 640)

    def validate(self):
        assert self.camera_cfg.buffer_size_hw == self.buffer_size_hw


class RenderScene(Scene):
    def __init__(self, cfg: RenderSceneConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.camera = Camera(self.cfg.camera_cfg)
        self.camera.cfg.buffer_size_hw = self.cfg.buffer_size_hw
        self.objects = [RenderObject(e) for e in self.cfg.objects_cfg]
        self.lights = [e for e in self.objects if np.sum(np.abs(e.material.cfg.emittance)) > 0]

    @staticmethod
    def intersect_ray_with_objects_list(ray, objects, objects_to_ignore=[]):
        depth = np.inf
        intersected_obj = None
        point = None
        for obj in objects:
            if obj in objects_to_ignore:
                continue
            intersection = obj.geometry.intersect_with_ray(ray)
            if intersection is not None:
                curr_depth = np.linalg.norm(ray.cfg.origin - intersection)
                if curr_depth < depth:
                    intersected_obj = obj
                    depth = curr_depth
                    point = intersection
        return intersected_obj, point

    def intersect_ray_with_scene_objects(self, ray, objects_to_ignore=[]):
        return RenderScene.intersect_ray_with_objects_list(ray, self.objects,objects_to_ignore)

    def intersect_ray_with_scene_lights(self, ray, objects_to_ignore=[]):
        return RenderScene.intersect_ray_with_objects_list(ray, self.lights, objects_to_ignore)


class PhongSceneConfig(RenderSceneConfig):
    def __init__(self):
        super().__init__()
        self.ambient_light = [1.0, 1.0, 1.0]


class PhongScene(RenderScene):
    def __init__(self, cfg: PhongSceneConfig):
        super().__init__(cfg)
        self.cfg = cfg
        for o in self.objects:
            o.material.cfg.ambient = self.cfg.ambient_light
