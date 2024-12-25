from abc import ABC, abstractmethod
from objects.config import Config
from objects.ray import Ray
import taichi as ti
import numpy as np

class GeometryConfig(Config):
    def __init__(self):
        super().__init__()
        self.transform = None  # TODO: implement


class Geometry(ABC):
    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg

    @abstractmethod
    def intersect_with_ray(self, ray: ti.template()):
        pass

    @abstractmethod
    def get_normal_at_point(self, point: ti.types.vector(3,ti.f32)):
        # if point is not on object - garbage out
        pass


class SphereConfig(GeometryConfig):
    def __init__(self):
        super().__init__()
        self.radius = 1.0
        self.origin = np.array([0.0, 0.0, 0.0])

    def validate(self):
        assert self.radius > 0


class Sphere(Geometry):
    def __init__(self, cfg: SphereConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def intersect_with_ray(self, ray:Ray):

        assert np.isclose(np.linalg.norm(ray.cfg.dir),1)

        o = ray.cfg.origin
        d = ray.cfg.dir
        c = self.cfg.origin
        r = self.cfg.radius

        delta_c = o - c
        dc_dot_d = delta_c.dot(d)
        discriminant = dc_dot_d - (dc_dot_d * delta_c.dot(d) - r ** 2)
        t = np.nan
        if discriminant == 0 and -dc_dot_d > 0:  # TODO: near/far clipping
            t = -dc_dot_d
        elif discriminant > 0:
            sqrt_discriminant = ti.sqrt(discriminant)
            t_1 = -dc_dot_d + sqrt_discriminant
            t_2 = -dc_dot_d - sqrt_discriminant
            if min(t_1, t_2) > 0:
                t = min(t_1, t_2)
            elif max(t_1, t_2) > 0:
                t = max(t_1, t_2)

        if t is not None:
            return o + t * d
        else:
            return None

    def get_normal_at_point(self, point):
        n = point - self.cfg.origin
        n /= np.linalg.norm(n)
        return n


class PlaneConfig(GeometryConfig):
    def __init__(self):
        super().__init__()
        self.normal = [0.0, 0.0, -1.0]
        self.is_bidirectional = False


class CubeConfig(GeometryConfig):
    def __init__(self):
        super().__init__()
        self.side_length = 1


geometry_factory = {
    SphereConfig:Sphere
}


GEOMETRY_MAPPING ={
    SphereConfig:Sphere,
}