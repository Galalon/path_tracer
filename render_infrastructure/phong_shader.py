from objects.scene import PhongScene, PhongSceneConfig
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import copy  # To duplicate the scene

def preprocess(cfg: PhongSceneConfig):
    scene = PhongScene(cfg)
    return scene

def calc_phong_color_multiprocess(scene, buffer, num_processes=7):
    """
    Multithreaded Phong color calculation with scene duplication.
    Each process gets a duplicate of the scene and operates on its assigned rows.
    """
    height, _ = scene.cfg.buffer_size_hw
    rows_per_process = height // num_processes

    # Prepare arguments for each process
    args = []
    for p in range(num_processes):
        start_row = p * rows_per_process
        end_row = (p + 1) * rows_per_process if p != num_processes - 1 else height

        # Duplicate the scene for each process
        duplicated_scene = copy.deepcopy(scene)

        # Create a buffer for this process's results
        local_buffer = np.zeros_like(buffer)

        args.append((duplicated_scene, local_buffer, start_row, end_row))

    # Use multiprocessing to parallelize
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(calc_phong_color_row, *arg) for arg in args]

        # Collect results from each process
        results = [future.result() for future in futures]

    # Combine results into the main buffer
    for local_buffer in results:
        buffer += local_buffer

    return buffer

from tqdm import tqdm

def calc_phong_color_row(scene: PhongScene, buffer, start_row=0, end_row=480):
    height, width = scene.cfg.buffer_size_hw
    for i in tqdm(range(start_row, end_row)):
        for j in range(width):
            ray = scene.camera.cast_ray(j, i)
            depth = np.inf
            intersected_obj = None
            point = None
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
                diffuse = np.array([0.0, 0.0, 0.0])
                specular = np.array([0.0, 0.0, 0.0])
                for l in scene.lights:
                    if l is intersected_obj:
                        continue
                    light_ray = calc_light_dir(l, point, intersected_obj, scene)
                    curr_diffuse, curr_specular = intersected_obj.material.get_color_per_light(ray, light_ray,
                                                                                               point, n)
                    diffuse += curr_diffuse * l.material.cfg.emittance
                    specular += curr_specular * l.material.cfg.emittance
                color = intersected_obj.material.cfg.k_specular * specular + \
                        intersected_obj.material.cfg.k_diffuse * diffuse + \
                        intersected_obj.material.cfg.k_ambient * intersected_obj.material.cfg.ambient + \
                        intersected_obj.material.cfg.emittance
            else:
                color = scene.cfg.ambient_light
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

    downsample_factor = 1
    cfg = PhongSceneConfig()
    cfg.ambient_light = np.array([0.5, 0.6, 0.7])
    cfg.objects_cfg = []
    cfg.buffer_size_hw = (480 // downsample_factor, 640 // downsample_factor)
    light_cfg = RenderObjectConfig()
    light_cfg.geometry_cfg.transform.set_translation(4.0, 3.0, 3.0)
    light_cfg.material_cfg.emittance = 2*np.array([0.7, 0.8, 1.1])
    light_cfg.material_cfg.k_ambient = 0.1
    cfg.objects_cfg.append(light_cfg)

    cube_cfg = RenderObjectConfig()
    cube_cfg.geometry_cfg = CubeConfig()
    cube_cfg.geometry_cfg.transform.set_translation(5.0, 1.0, 1.0)
    cube_cfg.geometry_cfg.transform.apply_rotation(45, 'z')
    cube_cfg.material_cfg.diffuse = np.array([1.0, 0.0, 0.0])
    cube_cfg.material_cfg.specular = np.array([1.0, 1.0, 1.0])
    cube_cfg.material_cfg.glossiness = 7
    cube_cfg.material_cfg.k_ambient = 0.1
    cube_cfg.material_cfg.k_specular = 0.1
    cube_cfg.material_cfg.k_diffuse = 0.7

    cfg.objects_cfg.append(cube_cfg)

    sphere_cfg = RenderObjectConfig()
    sphere_cfg.geometry_cfg = SphereConfig()
    sphere_cfg.geometry_cfg.transform.set_scale(2, 2, 2)
    sphere_cfg.geometry_cfg.transform.set_translation(6.0, -4.0, 2.0)
    sphere_cfg.material_cfg.diffuse = np.array([0.0, 1.0, 0.0])
    sphere_cfg.material_cfg.specular = np.array([1.0, 1.0, 1.0])
    sphere_cfg.material_cfg.glossiness = 7
    sphere_cfg.material_cfg.k_ambient = 0.1
    sphere_cfg.material_cfg.k_specular = 0.5
    sphere_cfg.material_cfg.k_diffuse = 0.5
    cfg.objects_cfg.append(sphere_cfg)

    plane_config = RenderObjectConfig()
    plane_config.geometry_cfg = PlaneConfig()
    plane_config.geometry_cfg.transform.set_translation(0, 0, 0)
    plane_config.material_cfg.diffuse = np.array([1.0, 1.0, 0.5])
    plane_config.material_cfg.k_ambient = 0.3
    plane_config.material_cfg.k_diffuse = 0.5
    plane_config.material_cfg.k_specular = 0.5
    cfg.objects_cfg.append(plane_config)

    cfg.camera_cfg.transform.set_translation(-1, 0, 3.0)
    cfg.camera_cfg.transform.set_rotation_order('zxz')
    cfg.camera_cfg.transform.set_rotation(0, -80, 90)
    cfg.to_file(r'F:\Users\User\PycharmProjects\path_tracer\path_tracer\render_infrastructure\test_scene_1.json')
    cfg.camera_cfg.buffer_size_hw = (480 // downsample_factor, 640 // downsample_factor)
    image = render_pipeline(cfg=cfg,
                            preprocess=preprocess,
                            render_cpu=calc_phong_color_multiprocess,
                            render_gpu=None,
                            postprocess= postprocess,
                            debug=True,
                            n_channels=3)

    # Display the final image
    # plt.imshow(np.squeeze(image_normal))
    # plt.axis("off")
    # plt.show()

    # plt.title('depth map')
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.axis("off")

    plt.show()
