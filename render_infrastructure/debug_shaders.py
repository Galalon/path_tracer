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
            for obj_idx, obj in enumerate(scene.objects):
                intersection = obj.geometry.intersect_with_ray(ray)
                if intersection is not None:
                    is_intersect = True
                    break
            buffer[i, j] = (obj_idx + 1) * int(is_intersect)
    return buffer


def calc_depth(scene: RenderScene, buffer):
    height, width = scene.cfg.buffer_size_hw
    for i in range(height):
        for j in range(width):
            ray = scene.camera.cast_ray(j, i)
            depth = np.inf
            for obj in scene.objects:
                intersection = obj.geometry.intersect_with_ray(ray)
                if intersection is not None:
                    curr_depth = np.linalg.norm(ray.cfg.origin - intersection)
                    depth = min(depth, curr_depth)
            buffer[i, j] = depth
    return buffer


def calc_normal(scene: RenderScene, buffer):
    height, width = scene.cfg.buffer_size_hw
    for i in range(height):
        for j in range(width):
            if i == 240 and j == 320:
                i = i
            ray = scene.camera.cast_ray(j, i)
            depth = np.inf
            normal = np.array([0, 0, 0])

            for obj in scene.objects:
                intersection = obj.geometry.intersect_with_ray(ray)
                if intersection is not None:
                    curr_depth = np.linalg.norm(ray.cfg.origin - intersection)
                    if curr_depth < depth:
                        normal = obj.geometry.get_normal_at_point(intersection)
                        depth = curr_depth
            buffer[i, j] = normal
    return buffer


if __name__ == "__main__":
    from render_infrastructure.render_pipeline import render_pipeline
    from objects.render_object import RenderObject, RenderObjectConfig
    from objects.primitive_geometry import *
    from objects.transform import Transform

    cfg = RenderSceneConfig()
    cfg.objects_cfg = []
    sphere_cfg = SphereConfig()
    sphere_cfg.transform.apply_translation(-2, 'z')
    sphere_cfg.transform.apply_scale(0.5, 'x')

    sphere_cfg_ = RenderObjectConfig()
    sphere_cfg_.geometry_cfg = sphere_cfg
    cfg.objects_cfg.append(sphere_cfg_)

    plane_config = PlaneConfig()
    plane_config.transform.apply_translation(-2, 'z')
    plane_config.transform.apply_rotation(70, 'x')

    plane_config_ = RenderObjectConfig()
    plane_config_.geometry_cfg = plane_config
    cfg.objects_cfg.append(plane_config_)

    cube_config = CubeConfig()
    cube_config.transform.set_translation(2, -1, -2.5)
    cube_config.transform.set_scale(2, 0.5, 0.5)
    cube_config.transform.set_rotation(0, 20, 60)

    cube_config_ = RenderObjectConfig()
    cube_config_.geometry_cfg = cube_config
    cfg.objects_cfg.append(cube_config_)

    cfg.camera_cfg.transform.apply_translation(-1, 'y')
    cfg.camera_cfg.transform.apply_translation(1, 'z')
    cfg.camera_cfg.transform.apply_rotation(10, 'x')
    cfg.camera_cfg.transform.apply_rotation(7.5, 'y')
    cfg.camera_cfg.transform.apply_rotation(3, 'z')
    image_depth = render_pipeline(cfg=cfg,
                                  preprocess=preprocess,
                                  render_cpu=calc_depth,
                                  render_gpu=None,
                                  postprocess=lambda s, b: np.minimum(
                                      np.log(np.where(np.isinf(b), np.nanmax(b[np.isfinite(b)]) * 2, b)), 4),
                                  debug=True,
                                  n_channels=1)
    # # Display the final image
    # plt.imshow(np.squeeze(image_depth), cmap='gray')
    # plt.axis("off")
    # plt.savefig("mandelbrot.png", dpi=300)
    # plt.show()

    image_normal = render_pipeline(cfg=cfg,
                                   preprocess=preprocess,
                                   render_cpu=calc_normal,
                                   render_gpu=None,
                                   postprocess=lambda s, b: b / 2 + 0.5,
                                   debug=True,
                                   n_channels=3)
    # Display the final image
    # plt.imshow(np.squeeze(image_normal))
    # plt.axis("off")
    # plt.show()
    plt.subplot(121)
    plt.title('depth map')
    plt.imshow(np.squeeze(image_depth), cmap='gray')
    plt.axis("off")
    plt.subplot(122)
    plt.title('normal map')
    plt.imshow(np.squeeze(image_normal), cmap='gray')
    plt.axis("off")
    plt.show()
