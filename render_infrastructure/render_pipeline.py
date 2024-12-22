import numpy as np
import time
import taichi as ti
from matplotlib import pyplot as plt
from objects.scene import Config


# Main function
def render_pipeline(cfg: Config, preprocess, render_cpu=None, render_gpu=None, postprocess=None, debug=False):
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
        buffer = np.zeros(scene.cfg.buffer_size, dtype=np.float32)
        raw_buffer = render_cpu(scene, buffer)  # Use CPU rendering for debugging
    else:
        # Allocate buffer locally
        buffer = ti.field(dtype=ti.f32, shape=scene.cfg.buffer_size)
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
    print('Running on gpu:')
    image_gpu = render_pipeline(cfg=cfg,
                                preprocess=preprocess,
                                render_cpu=render_cpu,
                                render_gpu=render_gpu,
                                postprocess=postprocess,
                                debug=False)
    print('Running on cpu:')
    image_cpu = render_pipeline(cfg=cfg,
                                preprocess=preprocess,
                                render_cpu=render_cpu,
                                render_gpu=render_gpu,
                                postprocess=postprocess,
                                debug=True)
    # Display the final image
    plt.subplot(121)
    plt.title('gpu')
    plt.imshow(image_gpu)
    plt.axis("off")
    plt.subplot(122)
    plt.title('cpu')
    plt.imshow(image_gpu)
    plt.axis("off")
    plt.show()
