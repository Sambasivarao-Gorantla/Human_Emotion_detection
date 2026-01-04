"""
Metrics computation and evaluation module
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class MetricsComputer:
    """Compute evaluation metrics"""
    
    @staticmethod
    def compute_metrics(eval_pred):
        """
        Compute accuracy from predictions and labels
        
        Args:
            eval_pred (tuple): (predictions_logits, true_labels)
        
        Returns:
            dict: Dictionary with accuracy metric
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    @staticmethod
    def compute_detailed_metrics(predictions, labels, class_names=None):
        """
        Compute comprehensive metrics (accuracy, precision, recall, F1)
        
        Args:
            predictions (array): Predicted class indices
            labels (array): True class labels
            class_names (list): Optional list of class names
        
        Returns:
            dict: Dictionary with all metrics
        """
        accuracy = accuracy_score(labels, predictions)
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        
        metrics = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro
        }
        
        # Per-class metrics if class names provided
        if class_names is not None:
            precision_per_class = precision_score(
                labels, predictions, average=None, zero_division=0
            )
            recall_per_class = recall_score(
                labels, predictions, average=None, zero_division=0
            )
            f1_per_class = f1_score(
                labels, predictions, average=None, zero_division=0
            )
            
            for idx, class_name in enumerate(class_names):
                if idx < len(precision_per_class):
                    metrics[f"precision_{class_name}"] = precision_per_class[idx]
                    metrics[f"recall_{class_name}"] = recall_per_class[idx]
                    metrics[f"f1_{class_name}"] = f1_per_class[idx]
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(predictions, labels):
        """
        Get confusion matrix
        
        Args:
            predictions (array): Predicted class indices
            labels (array): True class labels
        
        Returns:
            array: Confusion matrix
        """
        return confusion_matrix(labels, predictions)
    
    @staticmethod
    def print_metrics(metrics, class_names=None):
        """
        Pretty print metrics
        
        Args:
            metrics (dict): Dictionary of metrics
            class_names (list): Optional class names for per-class metrics
        """
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        # Overall metrics
        print(f"Accuracy:         {metrics.get('accuracy', 0):.4f}")
        print(f"Precision (macro): {metrics.get('precision_macro', 0):.4f}")
        print(f"Recall (macro):    {metrics.get('recall_macro', 0):.4f}")
        print(f"F1 Score (macro):  {metrics.get('f1_macro', 0):.4f}")
        
        # Per-class metrics
        if class_names is not None:
            print("\n" + "-"*60)
            print("PER-CLASS METRICS")
            print("-"*60)
            for class_name in class_names:
                precision = metrics.get(f"precision_{class_name}", 0)
                recall = metrics.get(f"recall_{class_name}", 0)
                f1 = metrics.get(f"f1_{class_name}", 0)
                print(f"{class_name:20s} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
        
        print("="*60 + "\n")
