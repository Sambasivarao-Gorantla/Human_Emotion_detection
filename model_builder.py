"""
Model initialization and setup module
"""

import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification
import config


class ModelBuilder:
    """Handles model loading, modification, and setup"""
    
    def __init__(self, num_classes, pretrained_model=config.PRETRAINED_MODEL, 
                 device=config.DEVICE):
        """
        Initialize model builder
        
        Args:
            num_classes (int): Number of output classes
            pretrained_model (str): Hugging Face model identifier
            device (str): Device to load model on ('cuda' or 'cpu')
        """
        self.num_classes = num_classes
        self.pretrained_model = pretrained_model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
    
    def load_model(self, freeze_backbone=config.FREEZE_BACKBONE):
        """
        Load pretrained model and modify for new number of classes
        
        Args:
            freeze_backbone (bool): Whether to freeze encoder layers
        
        Returns:
            model: Modified vision transformer model
        """
        print(f"Loading pretrained model: {self.pretrained_model}")
        self.model = AutoModelForImageClassification.from_pretrained(
            self.pretrained_model, 
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        if freeze_backbone:
            self._freeze_backbone()
        
        print(f"✓ Model loaded and moved to {self.device}")
        return self.model
    
    def _freeze_backbone(self):
        """Freeze encoder (ViT) layers, only train classifier"""
        if hasattr(self.model, 'vit'):
            for param in self.model.vit.parameters():
                param.requires_grad = False
            print("✓ ViT backbone frozen - only classifier will be trained")
        elif hasattr(self.model, 'vision_model'):
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            print("✓ Vision model backbone frozen - only classifier will be trained")
    
    def unfreeze_backbone(self, unfreeze_layers=None):
        """
        Unfreeze backbone layers for fine-tuning
        
        Args:
            unfreeze_layers (list): List of layer names to unfreeze. 
                                   If None, unfreezes all layers.
        """
        if hasattr(self.model, 'vit'):
            for param in self.model.vit.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'vision_model'):
            for param in self.model.vision_model.parameters():
                param.requires_grad = True
        print("✓ Backbone unfrozen")
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def print_model_info(self):
        """Print model architecture and parameter info"""
        trainable = self.get_trainable_params()
        total = self.get_total_params()
        print(f"\n{'='*60}")
        print(f"Model: {self.pretrained_model}")
        print(f"Device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {total - trainable:,}")
        print(f"{'='*60}\n")
    
    def get_model(self):
        """Get model instance"""
        if self.model is None:
            self.load_model()
        return self.model
