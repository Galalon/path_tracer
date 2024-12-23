from objects.config import Config
import numpy as np


class CameraConfig(Config):
    def __init__(self):
        super().__init__()
        self.buffer_size_hw = (480, 640)
        self.fov_x = 90  # in degrees
        self.fov_y = None  # as long as one fov is initialized the other will be determined from the resolution
        self.transform = None

    def validate(self):
        if self.fov_x is not None:
            assert 0 < self.fov_x < 180, 'fov x out of valid range'
        elif self.fov_y is not None:
            assert 0 < self.fov_y < 180, 'fov y out of valid range'
        else:
            raise ValueError("Both fov's were not initialized set at least one of them")
        if self.fov_x is not None and self.fov_y is not None:
            assert self.fov_x * self.buffer_size_hw[0] == self.fov_y * self.buffer_size_hw[1], \
                'fov aspect ratio does not match buffer aspect ratio - change one of the variables or set one of' \
                ' the fovs to None for automatic calculation'


class Camera:
    def __init__(self, cfg: CameraConfig):
        cfg.validate()
        self.cfg = cfg
        self.cfg.fov_x = self.cfg.fov_y * self.cfg.buffer_size_hw[1] / self.cfg.buffer_size_hw[0] \
            if self.cfg.fov_x is None else self.cfg.fov_x
        self.cfg.fov_y = self.cfg.fov_x * self.cfg.buffer_size_hw[0] / self.cfg.buffer_size_hw[1] \
            if self.cfg.fov_y is None else self.cfg.fov_y

    def cast_ray(self, x_p, y_p):
        height, width = self.cfg.buffer_size_hw
        # fencepost problem
        height -= 1
        width -= 1
        tan_fov_x = np.tan(np.deg2rad(self.cfg.fov_x / 2))
        tan_fov_y = np.tan(np.deg2rad(self.cfg.fov_y / 2))
        ray_dir = [(2 * x_p - width) / width * tan_fov_x,
                   (2 * y_p - height) / height * tan_fov_y,
                   -1]
        ray_origin = [0, 0, 0]
        return ray_dir, ray_origin  # TODO: ray class


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def plot_3d_vector_field(points, directions, arrow_length=0.1, normalize=True, color='blue'):
        """
        Plots a 3D vector field from points and direction vectors.

        Parameters:
        - points (ndarray): Array of shape (n, 3), 3D positions.
        - directions (ndarray): Array of shape (n, 3), direction vectors.
        - arrow_length (float): Length of arrows.
        - normalize (bool): Normalize vectors to unit length if True.
        - color (str): Color of the arrows.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(*points.T, *directions.T, length=arrow_length, normalize=normalize, color=color)
        ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')


    cfg = CameraConfig()
    cfg.buffer_size_hw = (3, 5)

    camera = Camera(cfg)
    dirs = []
    origins = []
    for i in range(cfg.buffer_size_hw[0]):
        for j in range(cfg.buffer_size_hw[1]):
            dir_, origin = camera.cast_ray(j, i)
            dirs.append(dir_)
            origins.append(origin)
    plot_3d_vector_field(np.array(origin), np.array(dirs))
    plt.show()
