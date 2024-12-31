from objects.scene import RenderScene, RenderSceneConfig
import matplotlib.pyplot as plt
import numpy as np


def preprocess(cfg: RenderSceneConfig):
    scene = RenderScene(cfg)
    return scene


def calc_phong_color(scene: RenderScene, buffer):
    height, width = scene.cfg.buffer_size_hw
    for i in range(height):
        for j in range(width):
            ray = scene.camera.cast_ray(j, i)
            depth = np.inf
            intersected_obj = None
            point = None
            color = [-1.0, -1.0, -1.0]
            for obj in scene.objects:
                intersection = obj.geometry.intersect_with_ray(ray)
                if intersection is not None:
                    curr_depth = np.linalg.norm(ray.cfg.origin - intersection)
                    if curr_depth < depth:
                        intersected_obj = obj
                        depth = curr_depth
                        point = intersection
            if intersected_obj is not None:
                n = intersected_obj.geometry.get_normal_at_point(point)
                color = np.array([0.0, 0.0, 0.0])
                for l in scene.lights:
                    if l is intersected_obj:
                        continue
                    light_ray = calc_light_dir(l, point, intersected_obj, scene)
                    color += intersected_obj.material.get_color_per_light(ray, light_ray, intersection, n)
                color += intersected_obj.material.cfg.ambient + intersected_obj.material.cfg.emittance
            buffer[i, j] = color
    return buffer


from objects.ray import Ray, RayConfig


def calc_light_dir(l, point, intersected_obj, scene):
    l_dir = l.geometry.get_light_source_dir(point)
    light_ray_cfg = RayConfig()
    light_ray_cfg.origin = point.copy()
    light_ray_cfg.dir = l_dir
    light_ray = Ray(light_ray_cfg)
    light_intersection = l.geometry.intersect_with_ray(light_ray)
    light_intersection_dist = np.linalg.norm(light_intersection - point)
    assert light_intersection is not None
    for obj in scene.objects:  # TODO: intersection loop?
        if obj is not l and obj is not intersected_obj:
            intersection = obj.geometry.intersect_with_ray(light_ray)
            if intersection is not None and np.linalg.norm(intersection - point) < light_intersection_dist:
                return None
    return light_ray


def postprocess(scene, raw_buffer):
    return np.clip(raw_buffer, a_min=0, a_max=1)


if __name__ == "__main__":
    from render_infrastructure.render_pipeline import render_pipeline
    from objects.render_object import RenderObjectConfig
    from objects.primitive_geometry import *
    from objects.transform import Transform
    from render_infrastructure.debug_shaders import calc_depth

    downsample_factor = 5
    cfg = RenderSceneConfig()
    cfg.objects_cfg = []
    cfg.buffer_size_hw = (480 // downsample_factor, 640 // downsample_factor)
    sphere_cfg = RenderObjectConfig()
    sphere_cfg.geometry_cfg.transform.set_translation(5.0, 3.0, 1.0)
    sphere_cfg.material_cfg.emittance = np.array([1.0, 1.0, 1.0])

    cfg.objects_cfg.append(sphere_cfg)

    sphere_cfg = RenderObjectConfig()
    sphere_cfg.geometry_cfg.transform.set_translation(5.0, 0.0, 1.0)
    sphere_cfg.material_cfg.diffuse = np.array([1.0, 0.0, 0.0])
    sphere_cfg.material_cfg.specular = np.array([1.0, 1.0, 1.0])
    sphere_cfg.material_cfg.glossiness = 5

    cfg.objects_cfg.append(sphere_cfg)

    plane_config = RenderObjectConfig()
    plane_config.geometry_cfg = PlaneConfig()
    plane_config.geometry_cfg.transform.set_translation(0, 0, 0)
    plane_config.material_cfg.diffuse = np.array([1.0, 1.0, 0.5])
    cfg.objects_cfg.append(plane_config)

    cfg.camera_cfg.transform.set_translation(0, 0, 0.5)

    cfg.camera_cfg.transform.set_rotation(0, -90, 0)

    cfg.camera_cfg.buffer_size_hw = (480 // downsample_factor, 640 // downsample_factor)
    image = render_pipeline(cfg=cfg,
                            preprocess=preprocess,
                            render_cpu=calc_phong_color,
                            render_gpu=None,
                            postprocess=lambda s, b: b,
                            debug=True,
                            n_channels=3)

    # Display the final image
    # plt.imshow(np.squeeze(image_normal))
    # plt.axis("off")
    # plt.show()

    plt.title('depth map')
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.axis("off")

    plt.show()
