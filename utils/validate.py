"""
CAM generation and visualization utilities
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import ttach as tta
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from .evaluate import ConfusionMatrixAllClass
from .hierarchical_utils import merge_to_parent_predictions, merge_subclass_cams_to_parent
from .pyutils import AverageMeter
from .common import merge_multiscale_predictions, merge_multiscale_cams, compute_multiscale_loss, get_color_palette


def get_seg_label(cams, inputs, label):
    """Generate segmentation labels from CAMs"""
    with torch.no_grad():
        b, c, h, w = inputs.shape
        label = label.view(b, -1, 1, 1).cpu().data.numpy()
        cams = cams.cpu().data.numpy()
        cams = np.maximum(cams, 0)
        
        # Normalize CAMs to [0,1]
        channel_max = np.max(cams, axis=(2, 3), keepdims=True)
        channel_min = np.min(cams, axis=(2, 3), keepdims=True)
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6)
        cams = cams * label
        cams = torch.from_numpy(cams).float()
        
        # Resize to input dimensions
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=True)
        cam_max = torch.max(cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams, bg_cam], dim=1)

    return cams



def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    """Validation function with test-time augmentation"""
    model.eval()
    avg_meter = AverageMeter()
    fuse234_matrix = ConfusionMatrixAllClass(num_classes=cfg.dataset.num_classes + 1)
    
    # Test-time augmentation setup
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1])
    ])
    with torch.no_grad():
        for data in tqdm(data_loader,
                         total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data
            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list = model(inputs, )

            # Merge subclass predictions to parent class predictions
            cls1, cls2, cls3, cls4 = merge_multiscale_predictions(
                [cls1, cls2, cls3, cls4], k_list, method=cfg.train.merge_test)

            # Compute multi-scale weighted loss
            weights = (cfg.train.scale1_weight, cfg.train.scale2_weight,
                      cfg.train.scale3_weight, cfg.train.scale4_weight)
            cls_loss = compute_multiscale_loss([cls1, cls2, cls3, cls4], cls_label, cls_loss_func, weights)

            cls4 = (torch.sigmoid(cls4) > 0.5).float()
            all_cls_acc4 = (cls4 == cls_label).all(dim=1).float().sum() / cls4.shape[0] * 100
            avg_cls_acc4 = ((cls4 == cls_label).sum(dim=0) / cls4.shape[0]).mean() * 100
            avg_meter.add({"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4, "cls_loss": cls_loss})
            
            # Generate evaluation CAMs with TTA
            cams1 = []
            cams2 = []
            cams3 = []
            cams4 = []
            for tta_trans in tta_transform:
                augmented_tensor = tta_trans.augment_image(inputs)
                cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list = model(augmented_tensor)

                # Merge subclass CAMs and predictions to parent classes
                cam1, cam2, cam3, cam4 = merge_multiscale_cams(
                    [cam1, cam2, cam3, cam4], k_list, method=cfg.train.merge_test)
                cls1, cls2, cls3, cls4 = merge_multiscale_predictions(
                    [cls1, cls2, cls3, cls4], k_list, method=cfg.train.merge_test)

                cam1 = get_seg_label(cam1, augmented_tensor, cls_label).cuda()
                cam1 = tta_trans.deaugment_mask(cam1).unsqueeze(dim=0)
                cams1.append(cam1)
                cam2 = get_seg_label(cam2, augmented_tensor, cls_label).cuda()
                cam2 = tta_trans.deaugment_mask(cam2).unsqueeze(dim=0)
                cams2.append(cam2)
                cam3 = get_seg_label(cam3, augmented_tensor, cls_label).cuda()
                cam3 = tta_trans.deaugment_mask(cam3).unsqueeze(dim=0)
                cams3.append(cam3)
                cam4 = get_seg_label(cam4, augmented_tensor, cls_label).cuda()
                cam4 = tta_trans.deaugment_mask(cam4).unsqueeze(dim=0)
                cams4.append(cam4)

            cams1 = torch.cat(cams1, dim=0).mean(dim=0)
            cams2 = torch.cat(cams2, dim=0).mean(dim=0)
            cams3 = torch.cat(cams3, dim=0).mean(dim=0)
            cams4 = torch.cat(cams4, dim=0).mean(dim=0)

            # Fuse multi-scale predictions
            fuse234 = 0.3 * cams2 + 0.3 * cams3 + 0.4 * cams4
            fuse_label234 = torch.argmax(fuse234, dim=1).cuda()
            
            fuse234_matrix.update(labels.detach().clone(), fuse_label234.clone())

    all_cls_acc4, avg_cls_acc4, cls_loss = avg_meter.pop('all_cls_acc4'), avg_meter.pop("avg_cls_acc4"), avg_meter.pop(
        "cls_loss")
    fuse234_score = fuse234_matrix.compute()[2]
    model.train()
    return all_cls_acc4, avg_cls_acc4, fuse234_score, cls_loss


def generate_cam(model=None, data_loader=None, cfg=None, cls_loss_func=None):

    model.eval()

    with torch.no_grad():
        for data in tqdm(data_loader,
                         total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data

            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list = model(inputs, )

            # Merge subclass CAMs to parent class CAMs
            cam1, cam2, cam3, cam4 = merge_multiscale_cams(
                [cam1, cam2, cam3, cam4], k_list, method=cfg.train.merge_test)

            cam1 = get_seg_label(cam1, inputs, cls_label).cuda()
            cam2 = get_seg_label(cam2, inputs, cls_label).cuda()
            cam3 = get_seg_label(cam3, inputs, cls_label).cuda()
            cam4 = get_seg_label(cam4, inputs, cls_label).cuda()

            fuse234 = 0.3 * cam2 + 0.3 * cam3 + 0.4 * cam4
            output_fuse234 = torch.argmax(fuse234, dim=1).long()

            # Generate palette dynamically based on num_classes
            PALETTE = get_color_palette(cfg.dataset.num_classes, include_background=True)

            for i in range(len(output_fuse234)):
                pred_mask = Image.fromarray(output_fuse234[i].cpu().clone().squeeze().numpy().astype(np.uint8)).convert('P')
                flat_palette = [val for sublist in PALETTE for val in sublist]
                pred_mask.putpalette(flat_palette)
                pred_mask.save(os.path.join(cfg.output_dirs.pred_dir, name[i] + ".png"))
    model.train()
    return 
