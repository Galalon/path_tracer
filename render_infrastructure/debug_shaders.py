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
            ray = scene.camera.cast_ray(j,i)
            is_intersect = False
            for obj in scene.objects:
                intersection = obj.intersect_with_ray(ray)
                if intersection is not None:
                    is_intersect = True
                    break
            buffer[i, j] = is_intersect
        return buffer




if __name__ == "__main__":
    from render_infrastructure.render_pipeline import render_pipeline
    from objects.primitive_geometry import SphereConfig
    cfg = RenderSceneConfig()
    sphere_cfg = SphereConfig()
    sphere_cfg.origin = np.array([0.0, 0.0, -10.0])
    cfg.objects_cfg.append(sphere_cfg)
    image = render_pipeline(cfg=cfg,
                            preprocess=preprocess,
                            render_cpu=calc_intersection,
                            render_gpu=None,
                            postprocess=lambda s,b: b,
                            debug=True)
    # Display the final image
    plt.imshow(image)
    plt.axis("off")
    # plt.savefig("mandelbrot.png", dpi=300)
    plt.show()