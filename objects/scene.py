from abc import ABC
from objects.config import Config


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
        self.cfg = cfg #technically unnecesary but it fucks up the UI
        xmin, xmax, ymin, ymax = self.cfg.bbox
        self.cfg.resolution_y = self.cfg.resolution_x * (ymax - ymin) / (xmax - xmin)  # TODO: either one
        self.cfg.resolution_y = int(self.cfg.resolution_y)
        self.cfg.buffer_size_hw = (self.cfg.resolution_y, self.cfg.resolution_x)
