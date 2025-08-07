import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import torch
import json
import matplotlib.pyplot as plt


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')

def save_colormap_image(tensor, save_path, cmap='hot', show_colorbar=False):
    """
    Save a normalized tensor (0-1) as a color-mapped image.

    Args:
        tensor (torch.Tensor): 2D or 3D tensor (C, H, W) or (H, W). Must be in range [0, 1].
        save_path (str): Full path to save the image.
        cmap (str): Colormap to use (e.g., 'hot', 'jet', 'inferno', etc.).
        show_colorbar (bool): Whether to include a color bar in the saved image.
    """
    if tensor.dim() == 3:
        tensor = tensor.mean(0)  # (H, W)
    elif tensor.dim() != 2:
        raise ValueError("Expected 2D or 3D tensor")

    array = tensor.detach().cpu().numpy()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(array.shape[1] / 100.0, array.shape[0] / 100.0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    im = ax.imshow(array, cmap=cmap, vmin=0, vmax=1)

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=100, transparent=False)
    plt.close(fig)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def normalize_tensor(E, crop_border=4):
    """
    Normalize a PyTorch tensor to the range [0, 1] using min-max from center-cropped region.
    Edge pixels are normalized based on these statistics, ensuring they fall within [0, 1].

    Args:
        E (torch.Tensor): 2D, 3D (C,H,W), or 4D (N,C,H,W) tensor.
        crop_border (int): Number of pixels to crop from each edge before computing min/max.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    # Crop center region for robust min/max
    if E.dim() == 2:
        cropped = E[crop_border:-crop_border, crop_border:-crop_border]
    elif E.dim() == 3:
        cropped = E[:, crop_border:-crop_border, crop_border:-crop_border]
    elif E.dim() == 4:
        cropped = E[:, :, crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError("Unsupported tensor shape: expected 2D, 3D, or 4D tensor")

    min_val = cropped.min()
    max_val = cropped.max()

    # Normalize using min-max from cropped area
    E_norm = (E - min_val) / (max_val - min_val + 1e-8)

    # Clamp values to [0, 1] in case any outliers fall outside
    E_norm = E_norm.clamp(0.0, 1.0)

    return E_norm


def dict_save(dictdata, dictname):
    info_json = json.dumps(dictdata, separators=(',', ': '))
    f = open(dictname, 'w')
    f.write(info_json)
    f.close()

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg