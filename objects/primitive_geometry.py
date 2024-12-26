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
    def intersect_with_ray(self, ray: Ray):
        pass

    @abstractmethod
    def get_normal_at_point(self, point: np.ndarray):
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

    def intersect_with_ray(self, ray: Ray):

        assert np.isclose(np.linalg.norm(ray.cfg.dir), 1)

        o = ray.cfg.origin
        d = ray.cfg.dir
        c = self.cfg.origin
        r = self.cfg.radius

        delta_c = o - c
        dc_dot_d = delta_c.dot(d)
        discriminant = dc_dot_d ** 2 - (delta_c.dot(delta_c) - r ** 2)
        t = None
        if discriminant == 0 and -dc_dot_d > 0:  # TODO: near/far clipping
            t = -dc_dot_d
        elif discriminant > 0:
            sqrt_discriminant = np.sqrt(discriminant)
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
        self.normal = np.array([0.0, 0.0, -1.0])
        self.point = np.array([0.0, 0.0, 0.0])
        self.is_bidirectional = False

    def validate(self):
        assert np.linalg.norm(self.normal) > 1e-8, 'ill defined unit vector'
        return


class Plane(Geometry):
    def __init__(self, cfg: PlaneConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.cfg.normal /= np.linalg.norm(self.cfg.normal)

    def intersect_with_ray(self, ray: Ray):
        d = ray.cfg.dir
        o = ray.cfg.origin
        n = self.cfg.normal
        p = self.cfg.point

        d_dot_n = d.dot(n)
        if d_dot_n == 0:
            return None

        if not self.cfg.is_bidirectional and d_dot_n > 0:
            return None

        delta_po_dot_n = (p - o).dot(n)

        t = delta_po_dot_n / d_dot_n
        if t < 0:
            return None

        return o + t * d

    def get_normal_at_point(self, point: np.ndarray):
        return self.cfg.normal


class CubeConfig(GeometryConfig):
    def __init__(self):
        super().__init__()
        self.side_length = 1
        self.origin = np.array([0.0, 0.0, 0.0])

    def validate(self):
        assert self.side_length > 0


class Cube(Geometry):
    def __init__(self, cfg: CubeConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def intersect_with_ray(self, ray: Ray):
        d = ray.cfg.dir
        o = ray.cfg.origin
        c = self.cfg.origin
        r = self.cfg.side_length / 2

        parallel_intersection = (d == 0) & ((c - r > o) | (c + r < o))
        if parallel_intersection.any():
            return None

        t_min = (c - np.sign(d)*r) / d
        t_min[d == 0] = c[d == 0] - r
        t_min = np.max(t_min)
        t_max = (c + np.sign(d)*r) / d
        t_max[d == 0] = c[d == 0] + r
        t_max = np.min(t_max)
        if t_max < t_min or t_min < 0:
            return None
        return o + t_min * d

    def get_normal_at_point(self, point: np.ndarray):
        diff_fron_origin = point - self.cfg.origin
        ind_max = np.argmax(np.abs(diff_fron_origin))
        n = np.zeros(3)
        n[ind_max] = np.sign(diff_fron_origin[ind_max])
        return n


GEOMETRY_MAPPING = {
    SphereConfig: Sphere,
    PlaneConfig: Plane,
    CubeConfig: Cube,
}
