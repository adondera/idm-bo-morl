import torch
import numpy as np
import n_sphere

def reduce_dim(x: np.array or torch.tensor):
    """
    Project preference x to (n-1)-dimensional space
    """
    if type(x) == torch.Tensor:
        x = x.numpy()
    spherical_proj = n_sphere.convert_spherical(x)
    return spherical_proj[1:]


def increase_dim(x: dict or np.array):
    """
    Recover n-dimensional preference from (n-1)-dimensional preference
    """
    angles = x
    if type(x) == dict:
        angles = np.array(list(x.values()))
    # 1.0 is the norm/radius, x.values() are the angles
    l = torch.tensor(np.concatenate(([1.0], angles)), dtype=torch.float32)
    rectangular_proj = n_sphere.convert_rectangular(l)
    return torch.nn.functional.normalize(rectangular_proj, p=1.0, dim=0).numpy()