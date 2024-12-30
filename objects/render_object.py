from objects.primitive_geometry import GeometryConfig, Geometry, SphereConfig, GEOMETRY_MAPPING
from objects.material import PhongMaterialConfig, PhongMaterial
from objects.config import Config


class RenderObjectConfig(Config):
    def __init__(self):
        self.geometry_cfg = SphereConfig()  # dummy init
        self.material_cfg = PhongMaterialConfig()  # TODO: generalize

    def validate(self):
        assert isinstance(self.geometry_cfg, GeometryConfig)
        assert isinstance(self.material_cfg, PhongMaterialConfig)


class RenderObject:
    def __init__(self, cfg: RenderObjectConfig):
        self.cfg = cfg
        self.geometry = GEOMETRY_MAPPING[type(self.cfg.geometry_cfg)](self.cfg.geometry_cfg)
        self.material = PhongMaterial(self.cfg.material_cfg)
