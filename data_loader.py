"""
Data loading and preprocessing module
"""

import random
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor
import config


class DataProcessor:
    """Handles data loading, preprocessing, and splitting"""
    
    def __init__(self, dataset_path, processor_name=config.PRETRAINED_MODEL):
        """
        Initialize data processor
        
        Args:
            dataset_path (str): Path to dataset folder
            processor_name (str): Hugging Face model for image preprocessing
        """
        self.dataset_path = dataset_path
        self.processor = AutoImageProcessor.from_pretrained(processor_name, use_fast=True)
        self.dataset = None
        self.class_indices = None
        self.num_classes = None
    
    def transform(self, image):
        """Transform image using processor"""
        return self.processor(image, return_tensors='pt')['pixel_values'].squeeze(0)
    
    def load_dataset(self):
        """Load dataset from folder structure"""
        self.dataset = ImageFolder(self.dataset_path, transform=self.transform)
        self.num_classes = len(self.dataset.classes)
        print(f"✓ Dataset loaded: {len(self.dataset)} images, {self.num_classes} classes")
        return self.dataset
    
    def stratified_split(self, train_split=config.TRAIN_SPLIT, 
                        val_split=config.VAL_SPLIT, 
                        seed=config.RANDOM_SEED):
        """
        Stratified split maintaining class distribution
        
        Args:
            train_split (float): Proportion for training (0-1)
            val_split (float): Proportion for validation (0-1)
            seed (int): Random seed for reproducibility
        
        Returns:
            tuple: (train_indices, val_indices, test_indices)
        """
        if self.dataset is None:
            self.load_dataset()
        
        random.seed(seed)
        targets = self.dataset.targets
        class_indices = defaultdict(list)
        
        # Group indices by class
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
        
        self.class_indices = class_indices
        train_idx, val_idx, test_idx = [], [], []
        
        # Split per class
        for label, indices in class_indices.items():
            random.shuffle(indices)
            n = len(indices)
            n_train = int(train_split * n)
            n_val = int(val_split * n)
            
            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train:n_train + n_val])
            test_idx.extend(indices[n_train + n_val:])
        
        print(f"✓ Split completed - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        return train_idx, val_idx, test_idx
    
    def get_dataloaders(self, train_split=config.TRAIN_SPLIT, 
                       val_split=config.VAL_SPLIT,
                       batch_size=config.PER_DEVICE_BATCH_SIZE,
                       shuffle=config.SHUFFLE):
        """
        Create train, validation, and test dataloaders
        
        Args:
            train_split (float): Training proportion
            val_split (float): Validation proportion
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle training data
        
        Returns:
            dict: Dictionary with 'train', 'val', 'test' dataloaders
        """
        train_idx, val_idx, test_idx = self.stratified_split(train_split, val_split)
        
        train_data = Subset(self.dataset, train_idx)
        val_data = Subset(self.dataset, val_idx)
        test_data = Subset(self.dataset, test_idx)
        
        train_loader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=self.collate_fn
        )
        test_loader = DataLoader(
            test_data, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching"""
        images, labels = zip(*batch)
        return {
            "pixel_values": torch.stack(images),
            "labels": torch.tensor(labels)
        }
    
    def get_class_names(self):
        """Get class names from dataset"""
        if self.dataset is None:
            self.load_dataset()
        return self.dataset.classes
    
    def get_num_classes(self):
        """Get number of classes"""
        if self.num_classes is None:
            self.load_dataset()
        return self.num_classes
