import numpy as np
import taichi as ti
from matplotlib import pyplot as plt

from objects.scene import MandelbrotScene, MandelbrotConfig


# Preprocessing function to set up the scene
def preprocess(cfg: MandelbrotConfig):
    scene = MandelbrotScene(cfg)


    return scene


# GPU Rendering kernel
@ti.kernel
def render_gpu(scene: ti.template(), buffer: ti.template()):
    xmin, xmax, ymin, ymax = scene.cfg.bbox
    height, width = scene.cfg.buffer_size
    for i, j in buffer:  # Loop over all pixels
        # Map pixel (i, j) to complex number c
        x = xmin + (xmax - xmin) * j / width
        y = ymin + (ymax - ymin) * i / height
        c = ti.Vector([x, y])  # Represent complex number
        z = ti.Vector([0.0, 0.0])  # Initialize z = 0 + 0i
        iter_count = 0
        for k in range(scene.cfg.max_iters):
            # Correct complex squaring
            z_real = z[0] * z[0] - z[1] * z[1]  # Real part: a^2 - b^2
            z_imag = 2 * z[0] * z[1]  # Imaginary part: 2ab
            z_real += c[0]  # Add real part of c
            z_imag += c[1]  # Add imaginary part of c
            z = ti.Vector([z_real, z_imag])  # Update z
            if z.norm() > scene.cfg.radius:  # Escape condition
                break
            iter_count = k
        buffer[i, j] = iter_count / scene.cfg.max_iters  # Normalize iteration count


# CPU Rendering function
def render_cpu(scene, buffer):
    xmin, xmax, ymin, ymax = scene.cfg.bbox
    height, width = scene.cfg.buffer_size
    for i in range(height):
        for j in range(width):
            # Map pixel (i, j) to complex number c
            x = xmin + (xmax - xmin) * j / width
            y = ymin + (ymax - ymin) * i / height
            c = np.array([x, y])
            z = np.array([0.0, 0.0])
            iter_count = 0
            for k in range(scene.cfg.max_iters):
                z_real = z[0] * z[0] - z[1] * z[1]
                z_imag = 2 * z[0] * z[1]
                z_real += c[0]
                z_imag += c[1]
                z = np.array([z_real, z_imag])
                if np.linalg.norm(z) > scene.cfg.radius:
                    break
                iter_count = k
            buffer[j, i] = iter_count / scene.cfg.max_iters  # Normalize iteration count
    return buffer


# Postprocessing function
def postprocess(scene, raw_buffer):
    # Map normalized iteration counts to colors (grayscale in this example)
    raw_buffer[np.isclose(raw_buffer, (scene.cfg.max_iters - 1) / scene.cfg.max_iters)] = 0
    image = plt.cm.viridis(raw_buffer)[:, :, :3]  # Extract RGB channels
    return image


if __name__ == "__main__":
    from render_infrastructure.render_pipeline import render_pipeline
    cfg = MandelbrotConfig()
    cfg.bbox = (0.2, 0.5, -0.25, 0.25)
    image = render_pipeline(cfg=cfg,
                            preprocess=preprocess,
                            render_cpu=render_cpu,
                            render_gpu=render_gpu,
                            postprocess=postprocess,
                            debug=False)
    # Display the final image
    plt.imshow(image)
    plt.axis("off")
    # plt.savefig("mandelbrot.png", dpi=300)
    plt.show()
