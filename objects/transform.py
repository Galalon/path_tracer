import numpy as np
from objects.config import Config


class Transform(Config):
    def __init__(self):
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])  # Euler angles in degrees
        self.scale = np.array([1.0, 1.0, 1.0])
        self.matrix = np.eye(4)  # Identity matrix
        self.inverse_matrix = np.eye(4)

    def set_translation(self, x: float, y: float, z: float):
        self.translation = np.array([x, y, z])
        self._recalculate_matrix()

    def set_rotation(self, rx: float, ry: float, rz: float):
        self.rotation = np.array([rx, ry, rz])
        self._recalculate_matrix()

    def set_scale(self, sx: float, sy: float, sz: float):
        self.scale = np.array([sx, sy, sz])
        self._recalculate_matrix()

    @staticmethod
    def _axis_to_index(axis: str) -> int:
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis in axis_map:
            return axis_map[axis]
        else:
            raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")

    def apply_translation(self, d: float, axis: str):
        index = self._axis_to_index(axis)
        self.translation[index] += d
        self._recalculate_matrix()

    def apply_rotation(self, d: float, axis: str):
        index = self._axis_to_index(axis)
        self.rotation[index] += d
        self._recalculate_matrix()

    def apply_scale(self, d: float, axis: str):
        index = self._axis_to_index(axis)
        self.scale[index] *= d
        self._recalculate_matrix()

    def set_all(self, translation=None, rotation=None, scale=None):
        if translation is not None:
            self.translation = np.array(translation)
        if rotation is not None:
            self.rotation = np.array(rotation)
        if scale is not None:
            self.scale = np.array(scale)
        self._recalculate_matrix()

    def _recalculate_matrix(self):
        t_matrix = Transform.matrix_translation(*self.translation)
        r_matrix = Transform.matrix_rotation_euler(*self.rotation)
        s_matrix = Transform.matrix_scaling(*self.scale)

        # Combine matrices in scale -> rotate -> translate order
        self.matrix = t_matrix @ r_matrix @ s_matrix
        self.inverse_matrix = np.linalg.inv(self.matrix)

    @staticmethod
    def matrix_translation(tx: float, ty: float, tz: float) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, 3] = [tx, ty, tz]
        return mat

    @staticmethod
    def matrix_scaling(sx: float, sy: float, sz: float) -> np.ndarray:
        mat = np.eye(4)
        mat[0, 0] = sx
        mat[1, 1] = sy
        mat[2, 2] = sz
        return mat

    @staticmethod
    def matrix_rotation_euler(rx: float, ry: float, rz: float) -> np.ndarray:
        # Convert degrees to radians
        rx, ry, rz = np.radians([rx, ry, rz])

        # Rotation matrices for X, Y, Z axes
        cos, sin = np.cos, np.sin

        rx_matrix = np.array([
            [1, 0, 0, 0],
            [0, cos(rx), -sin(rx), 0],
            [0, sin(rx), cos(rx), 0],
            [0, 0, 0, 1]
        ])

        ry_matrix = np.array([
            [cos(ry), 0, sin(ry), 0],
            [0, 1, 0, 0],
            [-sin(ry), 0, cos(ry), 0],
            [0, 0, 0, 1]
        ])

        rz_matrix = np.array([
            [cos(rz), -sin(rz), 0, 0],
            [sin(rz), cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Combine rotations in Z -> Y -> X order
        return rz_matrix @ ry_matrix @ rx_matrix

    def to_dict(self):
        return {'translation': self.translation, 'rotation': self.rotation, 'scale': self.scale}

    @classmethod
    def from_dict(cls, config_dict: dict):
        t = Transform()
        t.set_all(config_dict['translation'], config_dict['rotation'], config_dict['scale'])

        return t

    def validate(self):
        return


def affine_transform(o, transform_matrix):
    o_h = np.append(o, 1.0)
    o_t = transform_matrix @ o_h
    o_t = o_t[:-1] / o_t[-1]
    return o_t
