import torch
import os
from dinov2.models import vision_transformer as vision_transformer_dinov2
import logging

_logger = logging.getLogger(__name__)

_WEIGHTS_DIR = "backbones/weights"
os.makedirs(_WEIGHTS_DIR, exist_ok=True)

def load(name):
    arch, patchsize = name.split("_")[-2], name.split("_")[-1]
    model = vision_transformer_dinov2.__dict__[f'vit_{arch}'](patch_size=int(patchsize), img_size=518,
                                                              block_chunks=0, init_values=1e-8,
                                                              interpolate_antialias=False,
                                                              interpolate_offset=0.1)

    if "v2" in name:
        if "reg" in name:
            if arch == "base":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb{patchsize}/dinov2_vitb{patchsize}_reg4_pretrain.pth")
            elif arch == "small":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits{patchsize}/dinov2_vits{patchsize}_reg4_pretrain.pth")
            elif arch == "large":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl{patchsize}/dinov2_vitl{patchsize}_reg4_pretrain.pth")
            else:
                raise ValueError("Invalid type of architecture. It must be either 'small' or 'base' or 'large.")
        else:
            if arch == "base":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb{patchsize}/dinov2_vitb{patchsize}_pretrain.pth")
            elif arch == "small":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits{patchsize}/dinov2_vits{patchsize}_pretrain.pth")
            else:
                raise ValueError("Invalid type of architecture. It must be either 'small' or 'base'.")

        state_dict = torch.load(ckpt_pth, map_location='cpu')

    model.load_state_dict(state_dict, strict=False)
    return model

def download_cached_file(url, check_hash=True, progress=True):
    from torch.hub import download_url_to_file, urlparse
    if isinstance(url, (list, tuple)):
        url, filename = url
    else:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
    cached_file = os.path.join(_WEIGHTS_DIR, filename)
    if not os.path.exists(cached_file):
        _logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file
