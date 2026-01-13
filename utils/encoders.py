"""
Encoder factory for flexible patch encoding in PBIP.

Supports multiple vision models for feature extraction:
- MedCLIP: Medical foundation model (default)
- Virchow2: Pathology-specific model (PAIGE-AI)
- DinoV3: Generic vision model (Meta)

All encoders are frozen and used for prototype feature extraction.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class EncoderFactory:
    """Factory for creating and loading different patch encoders."""
    
    @staticmethod
    def create_encoder(encoder_name: str = "medclip", 
                      input_size: int = 224, 
                      device: torch.device = None) -> Tuple[nn.Module, int]:
        """
        Create an encoder model.
        
        Args:
            encoder_name: One of 'medclip', 'virchow2', 'dinov3'
            input_size: Input image size (224 recommended)
            device: GPU device to load model on
            
        Returns:
            Tuple of (encoder_model, feature_dim)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        encoder_name = encoder_name.lower().strip()
        
        if encoder_name == "medclip":
            return EncoderFactory._create_medclip(device)
        elif encoder_name == "virchow2":
            return EncoderFactory._create_virchow2(input_size, device)
        elif encoder_name == "dinov3":
            return EncoderFactory._create_dinov3(device)
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}. "
                           f"Supported: medclip, virchow2, dinov3")
    
    @staticmethod
    def _create_medclip(device: torch.device) -> Tuple[nn.Module, int]:
        """
        Create MedCLIP encoder.
        
        Output features: 512-dim
        Model: Vision Transformer from MedCLIP (medical foundation model)
        """
        try:
            from medclip import MedCLIPModel, MedCLIPVisionModelViT
            
            print("Loading MedCLIP encoder (medical foundation model)...")
            clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
            clip_model = clip_model.to(device)
            clip_model.eval()
            
            # MedCLIP vision model outputs 512-dim features
            feature_dim = 512
            
            return clip_model, feature_dim
            
        except ImportError:
            raise ImportError("MedCLIP not installed. Install with: pip install medclip")
    
    @staticmethod
    def _create_virchow2(input_size: int = 224, device: torch.device = None) -> Tuple[nn.Module, int]:
        """
        Create Virchow2 encoder from PAIGE-AI.
        
        Output features: 1280-dim (class token only)
        Model: Vision Transformer trained on histopathology data
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            import timm
            
            print("Loading Virchow2 encoder (pathology-specific vision model)...")
            encoder = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
            encoder = encoder.to(device)
            encoder.eval()
            
            # Wrap in a module that extracts class token
            virchow2_model = Virchow2Wrapper(encoder)
            
            # Virchow2 outputs 1280-dim class token
            feature_dim = 1280
            
            return virchow2_model, feature_dim
            
        except ImportError:
            raise ImportError("timm not installed. Install with: pip install timm")
    
    @staticmethod
    def _create_dinov3(device: torch.device = None) -> Tuple[nn.Module, int]:
        """
        Create DinoV3 encoder from Meta.
        
        Output features: 1024-dim
        Model: Vision Transformer trained with DINOv3 self-supervised learning
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            import timm
            
            print("Loading DinoV3 encoder (self-supervised vision model)...")
            encoder = timm.create_model(
                "hf-hub:timm/vit_large_patch16_dinov3.lvd1689m",
                pretrained=True,
                num_classes=0  # Remove classification head
            )
            encoder = encoder.to(device)
            encoder.eval()
            
            # DinoV3 outputs 1024-dim features
            feature_dim = 1024
            
            return encoder, feature_dim
            
        except ImportError:
            raise ImportError("timm not installed. Install with: pip install timm")


class Virchow2Wrapper(nn.Module):
    """
    Wrapper for Virchow2 to extract class token output.
    
    Virchow2 outputs all tokens; we extract just the class token (1280-dim)
    for consistency with other encoders.
    """
    
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.device = next(encoder.parameters()).device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract Virchow2 class token.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            
        Returns:
            Class token features (B, 1280)
        """
        output = self.encoder(x)  # (B, num_tokens, 1280)
        class_token = output[:, 0]  # Extract class token (B, 1280)
        return class_token


def get_encoder_config(encoder_name: str) -> dict:
    """
    Get recommended configuration for an encoder.
    
    Returns dict with:
    - input_size: Recommended input image size
    - feature_dim: Output feature dimension
    - description: Human-readable description
    """
    encoder_name = encoder_name.lower().strip()
    
    configs = {
        "medclip": {
            "input_size": 224,
            "feature_dim": 512,
            "description": "Medical foundation model (CLIP trained on medical texts/images)",
            "papers": ["https://arxiv.org/abs/2112.02624"],
            "best_for": "General pathology, text-aligned features"
        },
        "virchow2": {
            "input_size": 224,
            "feature_dim": 1280,
            "description": "Pathology-specific Vision Transformer by PAIGE-AI",
            "papers": ["https://arxiv.org/abs/2404.23228"],
            "best_for": "Histopathology, state-of-the-art performance"
        },
        "dinov3": {
            "input_size": 224,
            "feature_dim": 1024,
            "description": "Self-supervised Vision Transformer by Meta (DINOv3)",
            "papers": ["https://arxiv.org/abs/2305.08243"],
            "best_for": "General computer vision, self-supervised learning"
        }
    }
    
    if encoder_name not in configs:
        raise ValueError(f"Unknown encoder: {encoder_name}")
    
    return configs[encoder_name]
