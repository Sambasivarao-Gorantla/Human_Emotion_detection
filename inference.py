"""
Inference utilities for making predictions on new images
"""

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


class Inferencer:
    """Handle inference on new images"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize inferencer
        
        Args:
            model_path (str): Path to saved model or HF model ID
            device (str): Device to load model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.id2label = None
    
    def load_model_and_processor(self):
        """Load model and processor from saved checkpoint"""
        print(f"Loading model from {self.model_path}")
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_path
        ).to(self.device)
        
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        
        # Get id2label mapping if available
        if hasattr(self.model.config, 'id2label'):
            self.id2label = self.model.config.id2label
        
        self.model.eval()
        print("✓ Model loaded successfully")
    
    def predict_single(self, image_path, top_k=5):
        """
        Predict class for single image
        
        Args:
            image_path (str): Path to image file
            top_k (int): Return top-k predictions
        
        Returns:
            dict: Prediction results with probabilities
        """
        if self.model is None:
            self.load_model_and_processor()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(image, return_tensors='pt').to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
        top_probs = top_probs[0].cpu().numpy()
        top_indices = top_indices[0].cpu().numpy()
        
        results = {
            'image_path': image_path,
            'predictions': []
        }
        
        for prob, idx in zip(top_probs, top_indices):
            label = self.id2label.get(str(idx), f'Class {idx}') if self.id2label else f'Class {idx}'
            results['predictions'].append({
                'label': label,
                'class_id': int(idx),
                'probability': float(prob)
            })
        
        return results
    
    def predict_batch(self, image_paths, top_k=5):
        """
        Predict classes for multiple images
        
        Args:
            image_paths (list): List of image file paths
            top_k (int): Return top-k predictions
        
        Returns:
            list: List of prediction results
        """
        results = []
        for img_path in image_paths:
            result = self.predict_single(img_path, top_k)
            results.append(result)
        return results
    
    def predict_from_pil(self, pil_image, top_k=5):
        """
        Predict from PIL Image object
        
        Args:
            pil_image: PIL Image object
            top_k (int): Return top-k predictions
        
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            self.load_model_and_processor()
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        inputs = self.processor(pil_image, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
        top_probs = top_probs[0].cpu().numpy()
        top_indices = top_indices[0].cpu().numpy()
        
        results = {'predictions': []}
        
        for prob, idx in zip(top_probs, top_indices):
            label = self.id2label.get(str(idx), f'Class {idx}') if self.id2label else f'Class {idx}'
            results['predictions'].append({
                'label': label,
                'class_id': int(idx),
                'probability': float(prob)
            })
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize inferencer
    inferencer = Inferencer(
        model_path="./results/checkpoint-100",
        device='cuda'
    )
    
    # Single image prediction
    result = inferencer.predict_single("path/to/image.jpg", top_k=3)
    print(result)
    
    # Batch prediction
    results = inferencer.predict_batch(
        ["image1.jpg", "image2.jpg", "image3.jpg"],
        top_k=3
    )
    for r in results:
        print(r)
