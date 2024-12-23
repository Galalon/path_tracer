import numpy as np
import time
import taichi as ti
from matplotlib import pyplot as plt
from objects.scene import SceneConfig


# Main function
def render_pipeline(cfg: SceneConfig, preprocess, render_cpu=None, render_gpu=None, postprocess=None, debug=False):
    assert (render_cpu is not None) or (
        not debug), 'CPU render function is not defined, change debug mode or provide render_cpu'
    assert (render_gpu is not None) or (
        debug), 'GPU render function is not defined, change debug mode or provide render_gpu'

    # Initialize Taichi (switch between CPU and GPU by changing ti.cpu or ti.gpu)
    ti.init(arch=ti.gpu if not debug else ti.cpu)
    # Preprocessing
    print("Preprocessing...")
    scene = preprocess(cfg)

    # Main rendering loop
    print("Rendering...")
    start_time = time.time()
    if debug:
        buffer = np.zeros(scene.cfg.buffer_size_hw, dtype=np.float32)
        raw_buffer = render_cpu(scene, buffer)  # Use CPU rendering for debugging
    else:
        # Allocate buffer locally
        buffer = ti.field(dtype=ti.f32, shape=scene.cfg.buffer_size_hw)
        render_gpu(scene, buffer)  # Use GPU kernel for rendering
        raw_buffer = buffer.to_numpy()  # Convert Taichi buffer to NumPy array
    elapsed_time = time.time() - start_time
    print(f"Rendering completed in {elapsed_time:.2f} seconds.")

    # Postprocessing
    print("Postprocessing...")
    image = postprocess(scene, raw_buffer)
    print('Done !')
    return image


if __name__ == "__main__":
    from render_infrastructure.mandelbrot import preprocess, render_cpu, render_gpu, postprocess
    from objects.scene import MandelbrotConfig

    cfg = MandelbrotConfig()
    # cfg.buffer_size_hw = ()
    print('Running on gpu:')
    time_gpu = time.time()
    image_gpu = render_pipeline(cfg=cfg,
                                preprocess=preprocess,
                                render_cpu=render_cpu,
                                render_gpu=render_gpu,
                                postprocess=postprocess,
                                debug=False)
    time_gpu = time.time() - time_gpu
    print('Running on cpu:')
    time_cpu = time.time()
    image_cpu = render_pipeline(cfg=cfg,
                                preprocess=preprocess,
                                render_cpu=render_cpu,
                                render_gpu=render_gpu,
                                postprocess=postprocess,
                                debug=True)
    time_cpu = time.time() - time_cpu

    improvement_ratio = np.sqrt(time_cpu / time_gpu)
    cfg.resolution_x = max(int(cfg.resolution_x / improvement_ratio), 1)

    print('Running on cpu with decreased computation:')
    time_cpu_decreased = time.time()
    image_cpu_decreased = render_pipeline(cfg=cfg,
                                          preprocess=preprocess,
                                          render_cpu=render_cpu,
                                          render_gpu=render_gpu,
                                          postprocess=postprocess,
                                          debug=True)
    time_cpu_decreased = time.time() - time_cpu_decreased

    # Display the final image
    plt.subplot(131)
    plt.title(f'gpu ({time_gpu:.2f} s)')
    plt.imshow(image_gpu)
    plt.axis("off")
    plt.subplot(132)
    plt.title(f'cpu ({time_cpu:.2f} s)')
    plt.imshow(image_cpu)
    plt.axis("off")
    plt.subplot(133)
    plt.title(f'cpu low resolution ({time_cpu_decreased:.2f} s:)')
    plt.imshow(image_cpu_decreased)
    plt.axis("off")
    plt.show()
