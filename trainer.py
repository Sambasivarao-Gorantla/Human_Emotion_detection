"""
Training module using Hugging Face Trainer
"""

from transformers import TrainingArguments, Trainer
from data_loader import DataProcessor
from model_builder import ModelBuilder
from metrics import MetricsComputer
import config


class TrainerBuilder:
    """Builds and manages HF Trainer for model training"""
    
    def __init__(self, model, train_dataloader, val_dataloader, 
                 output_dir=config.OUTPUT_DIR, learning_rate=config.LEARNING_RATE):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            output_dir (str): Directory to save outputs
            learning_rate (float): Learning rate
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.trainer = None
    
    def setup_training_args(self, num_epochs=config.NUM_EPOCHS,
                           batch_size=config.PER_DEVICE_BATCH_SIZE,
                           eval_strategy=config.EVAL_STRATEGY,
                           save_strategy=config.SAVE_STRATEGY,
                           save_total_limit=config.SAVE_TOTAL_LIMIT):
        """
        Setup training arguments
        
        Args:
            num_epochs (int): Number of training epochs
            batch_size (int): Per-device batch size
            eval_strategy (str): Evaluation strategy ('epoch', 'steps', 'no')
            save_strategy (str): Saving strategy ('epoch', 'steps', 'no')
            save_total_limit (int): Maximum number of checkpoints to save
        
        Returns:
            TrainingArguments: Configured training arguments
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            learning_rate=self.learning_rate,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
            load_best_model_at_end=config.LOAD_BEST_MODEL,
            metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
            greater_is_better=config.GREATER_IS_BETTER,
            remove_unused_columns=False,
            logging_steps=10,
            logging_dir='./logs',
        )
        return training_args
    
    def create_trainer(self, training_args=None):
        """
        Create HF Trainer instance
        
        Args:
            training_args (TrainingArguments): Pre-configured training args.
                                             If None, uses default setup.
        
        Returns:
            Trainer: Configured trainer instance
        """
        if training_args is None:
            training_args = self.setup_training_args()
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataloader.dataset,
            eval_dataset=self.val_dataloader.dataset,
            data_collator=DataProcessor.collate_fn,
            compute_metrics=MetricsComputer.compute_metrics
        )
        
        print("✓ Trainer created")
        return self.trainer
    
    def train(self):
        """Train the model"""
        if self.trainer is None:
            self.create_trainer()
        
        print("Starting training...")
        self.trainer.train()
        print("✓ Training completed")
    
    def evaluate(self, test_dataset):
        """
        Evaluate model on test dataset
        
        Args:
            test_dataset: Test dataset
        
        Returns:
            dict: Evaluation results
        """
        if self.trainer is None:
            self.create_trainer()
        
        print("Evaluating on test set...")
        results = self.trainer.evaluate(test_dataset)
        print("✓ Evaluation completed")
        return results
    
    def get_trainer(self):
        """Get trainer instance"""
        if self.trainer is None:
            self.create_trainer()
        return self.trainer
