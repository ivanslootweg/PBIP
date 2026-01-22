import argparse
import torch
import numpy as np
import random
import logging
import hashlib
from omegaconf import OmegaConf


def build_uid_from_config(cfg):
    """Derive UID from config for prototype selection and training runs.
    
    UID format: {n_slides}_{k_list}_{nk}_{threshold_tag}_{selection_method}_{hash}
    Example: 500_k5-5_nk5_th0-9980-1-9980_top_attention_b34ecc4f
    """
    slides = getattr(cfg.features, 'n_slides_per_class_for_prototypes', None) if hasattr(cfg, 'features') else None
    selection = getattr(cfg.features, 'selection_method', None) if hasattr(cfg, 'features') else None
    selection = selection or 'all'
    
    # Cluster parameters
    k_list = getattr(cfg.features, 'k_list', None) if hasattr(cfg, 'features') else None
    nk = getattr(cfg.features, 'nk', None) if hasattr(cfg, 'features') else None
    k_str = "k" + "-".join(map(str, k_list)) if k_list is not None else "knone"
    nk_str = f"nk{nk}" if nk is not None else "nknone"

    threshold_cfg = None
    if hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'threshold_for_prototype'):
        threshold_cfg = cfg.dataset.threshold_for_prototype
        # Convert OmegaConf to native Python types
        if threshold_cfg is not None:
            threshold_cfg = OmegaConf.to_container(threshold_cfg, resolve=True)

    slides_str = str(slides) if slides is not None else 'all'

    # Normalize threshold into a stable tag (avoid curly braces for filenames)
    def _format_val(v):
        if v is None:
            return "none"
        try:
            return f"{float(v):.4f}".replace('.', '')  # e.g., 0.9980 -> 9980
        except Exception:
            return str(v).lower()

    if isinstance(threshold_cfg, dict):
        # Sort by key, format as key_val_key_val_...
        parts = [f"{k}{_format_val(threshold_cfg[k])}" for k in sorted(threshold_cfg.keys())]
        thresh_tag = "th" + "-".join(parts)
    elif isinstance(threshold_cfg, (list, tuple)):
        parts = [_format_val(v) for v in threshold_cfg]
        thresh_tag = "th" + "-".join(parts)
    elif threshold_cfg is not None:
        thresh_tag = "th" + _format_val(threshold_cfg)
    else:
        thresh_tag = "thall"

    # Include a short hash for stability
    cfg_fingerprint = hashlib.md5(str(getattr(cfg, 'dataset', '')).encode()).hexdigest()[:8]
    return f"{slides_str}_{k_str}_{nk_str}_{thresh_tag}_{selection}_{cfg_fingerprint}"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(filename='test.log'):
    ## setup logger
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]
        # Simple averaging for single value tracking
        self._sum = 0.0
        self._count = 0

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1
    
    def update(self, val, n=1):
        """Update with a single value (for simple averaging)."""
        self._sum += val * n
        self._count += n
    
    @property
    def avg(self):
        """Return average of tracked values."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v
