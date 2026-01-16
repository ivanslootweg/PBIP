import pickle as pkl
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from model.segform import mix_transformer



class AdaptiveLayer(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ClsNetwork(nn.Module):
    def __init__(self,
                 backbone='mit_b1',
                 num_classes=4,
                 stride=[4, 2, 2, 1],
                 pretrained=True,
                 n_ratio=0.5,
                 l_fea_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride

        # Use custom MixTransformer from segform module
        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        # Initialize encoder with pretrained weights if available
        if pretrained:
            pretrained_path = f'./pretrained/{backbone}.pth'
            if os.path.exists(pretrained_path):
                # weights_only=False needed for PyTorch 2.6+ compatibility
                state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
                state_dict.pop('head.weight', None)
                state_dict.pop('head.bias', None)
                state_dict = {k: v for k, v in state_dict.items() if k in self.encoder.state_dict().keys()}
                self.encoder.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights from {pretrained_path}")
            else:
                print(f"Warning: Pretrained weights not found at {pretrained_path}, using random initialization")

        self.pooling = F.adaptive_avg_pool2d

        ## medclip
        self.l_fc1 = AdaptiveLayer(512, n_ratio, self.in_channels[0])
        self.l_fc2 = AdaptiveLayer(512, n_ratio, self.in_channels[1])
        self.l_fc3 = AdaptiveLayer(512, n_ratio, self.in_channels[2])
        self.l_fc4 = AdaptiveLayer(512, n_ratio, self.in_channels[3])

        # Resolve label feature path: allow full .pkl path or basename
        # Supports both legacy and UID-based filenames
        resolved_path = None
        if isinstance(l_fea_path, str):
            # If it's an existing file path
            if os.path.isfile(l_fea_path):
                resolved_path = l_fea_path
            else:
                # Try as full path string ending with .pkl
                if l_fea_path.endswith('.pkl') and os.path.isfile(l_fea_path):
                    resolved_path = l_fea_path
                else:
                    # Try to find UID-based version
                    import glob
                    base_name = os.path.basename(l_fea_path)
                    if base_name.endswith('.pkl'):
                        base_name = base_name[:-4]
                    
                    # Search for UID-based file in same directory
                    dir_path = os.path.dirname(l_fea_path) or "./features/image_features"
                    uid_pattern = os.path.join(dir_path, f"{base_name}_*.pkl")
                    uid_files = glob.glob(uid_pattern)
                    
                    if uid_files:
                        resolved_path = uid_files[0]
                    else:
                        # Fallback to legacy location by basename
                        candidate = os.path.join("./features/image_features", f"{base_name}.pkl")
                        if os.path.isfile(candidate):
                            resolved_path = candidate
        if resolved_path is None:
            raise FileNotFoundError(f"Label feature .pkl not found. Provided l_fea_path='{l_fea_path}'. Tried UID-based, direct path, and ./features/image_features/{{name}}.pkl")

        with open(resolved_path, "rb") as lf:
            info = pkl.load(lf)
            self.l_fea = info['features'].cpu()
            self.k_list = info['k_list']
            self.nk = info.get('nk', 1)  # Nk representative images per subclass (paper uses 5)
            self.cumsum_k = info['cumsum_k']
            
            print(f"Loaded label features:")
            print(f"  Feature shape: {self.l_fea.shape}")
            print(f"  K (subclasses per class): {self.k_list}")
            print(f"  Nk (representatives per subclass): {self.nk}")
            print(f"  Total features stored: {self.l_fea.shape[0]} = {sum(self.k_list)} × {self.nk}")
            
        self.total_classes = sum(self.k_list) * self.nk  # K * Nk total features
        self.logit_scale1 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale2 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale3 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale4 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)


    def get_param_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    def forward(self, x):
        # MixTransformer returns (feature_maps, attentions)
        _x, _attns = self.encoder(x)
        
        # Ensure we have exactly 4 feature maps
        if len(_x) != 4:
            raise ValueError(f"Expected 4 feature maps from encoder, got {len(_x)}")

        logit_scale1 = self.logit_scale1
        logit_scale2 = self.logit_scale2
        logit_scale3 = self.logit_scale3
        logit_scale4 = self.logit_scale4

        imshape = [_.shape for _ in _x]
        image_features = [_.permute(0, 2, 3, 1).reshape(-1, _.shape[1]) for _ in _x]   
        _x1, _x2, _x3, _x4 = image_features
    

        l_fea = self.l_fea.to(x.device)
        l_fea1 = self.l_fc1(l_fea)
        l_fea2 = self.l_fc2(l_fea)
        l_fea3 = self.l_fc3(l_fea)
        l_fea4 = self.l_fc4(l_fea)
        
        _x1 = _x1 / _x1.norm(dim=-1, keepdim=True)
        logits_per_image1 = logit_scale1 * _x1 @ l_fea1.t().float() 
        out1 = logits_per_image1.view(imshape[0][0], imshape[0][2], imshape[0][3], -1).permute(0, 3, 1, 2) 
        cam1 = out1.clone().detach()
        cls1 = self.pooling(out1, (1, 1)).view(-1, self.total_classes)

        _x2 = _x2 / _x2.norm(dim=-1, keepdim=True)
        logits_per_image2 = logit_scale2 * _x2 @ l_fea2.t().float() 
        out2 = logits_per_image2.view(imshape[1][0], imshape[1][2], imshape[1][3], -1).permute(0, 3, 1, 2) 
        cam2 = out2.clone().detach()
        cls2 = self.pooling(out2, (1, 1)).view(-1, self.total_classes)

        _x3 = _x3 / _x3.norm(dim=-1, keepdim=True)
        logits_per_image3 = logit_scale3 * _x3 @ l_fea3.t().float() 
        out3 = logits_per_image3.view(imshape[2][0], imshape[2][2], imshape[2][3], -1).permute(0, 3, 1, 2) 
        cam3 = out3.clone().detach()
        cls3 = self.pooling(out3, (1, 1)).view(-1, self.total_classes)

        _x4 = _x4 / _x4.norm(dim=-1, keepdim=True)
        logits_per_image4 = logit_scale4 * _x4 @ l_fea4.t().float() 
        out4 = logits_per_image4.view(imshape[3][0], imshape[3][2], imshape[3][3], -1).permute(0, 3, 1, 2) 
        cam4 = out4.clone()
        cls4 = self.pooling(out4, (1, 1)).view(-1, self.total_classes)

        return cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, self.k_list, self.nk
