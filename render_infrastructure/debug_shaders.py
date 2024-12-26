import numpy as np
from objects.scene import RenderScene, RenderSceneConfig
import matplotlib.pyplot as plt


def preprocess(cfg: RenderSceneConfig):
    scene = RenderScene(cfg)
    return scene


def calc_intersection(scene: RenderScene, buffer):
    height, width = scene.cfg.buffer_size_hw
    for i in range(height):
        for j in range(width):
            ray = scene.camera.cast_ray(j, i)
            is_intersect = False
            for obj in scene.objects:
                intersection = obj.intersect_with_ray(ray)
                if intersection is not None:
                    is_intersect = True
                    break
            buffer[i, j] = is_intersect
    return buffer


def calc_depth(scene: RenderScene, buffer):
    height, width = scene.cfg.buffer_size_hw
    for i in range(height):
        for j in range(width):
            ray = scene.camera.cast_ray(j, i)
            depth = np.inf
            for obj in scene.objects:
                intersection = obj.intersect_with_ray(ray)
                if intersection is not None:
                    curr_depth = np.linalg.norm(ray.cfg.origin - intersection)
                    depth = min(depth, curr_depth)
            buffer[i, j] = depth
    return buffer


def calc_normal(scene: RenderScene, buffer):
    height, width = scene.cfg.buffer_size_hw
    for i in range(height):
        for j in range(width):
            ray = scene.camera.cast_ray(j, i)
            depth = np.inf
            normal = np.array([0, 0, 0])
            for obj in scene.objects:
                intersection = obj.intersect_with_ray(ray)
                if intersection is not None:
                    curr_depth = np.linalg.norm(ray.cfg.origin - intersection)
                    if curr_depth < depth:
                        normal = obj.get_normal_at_point(intersection)
                        depth = curr_depth
            buffer[i, j] = normal
    return buffer


if __name__ == "__main__":
    from render_infrastructure.render_pipeline import render_pipeline
    from objects.primitive_geometry import *

    cfg = RenderSceneConfig()
    sphere_cfg = SphereConfig()
    sphere_cfg.origin = np.array([0.0, 0.0, -2.0])

    cfg.objects_cfg.append(sphere_cfg)

    plane_config = PlaneConfig()
    plane_config.normal = np.array([0.0, -3.0, 1.0])
    plane_config.point = [0, 0, -2]

    cfg.objects_cfg.append(plane_config)

    cube_config = CubeConfig()
    cube_config.origin = np.array([2, -1, -2.5])
    cube_config.side_length = 0.5

    cfg.objects_cfg.append(cube_config)

    image = render_pipeline(cfg=cfg,
                            preprocess=preprocess,
                            render_cpu=calc_depth,
                            render_gpu=None,
                            postprocess=lambda s, b: np.log(b),
                            debug=True,
                            n_channels=1)
    # Display the final image
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.axis("off")
    # plt.savefig("mandelbrot.png", dpi=300)
    plt.show()

    image = render_pipeline(cfg=cfg,
                            preprocess=preprocess,
                            render_cpu=calc_normal,
                            render_gpu=None,
                            postprocess=lambda s, b: b/2 + 0.5,
                            debug=True,
                            n_channels=3)
    # Display the final image
    plt.imshow(np.squeeze(image))
    plt.axis("off")
    plt.show()