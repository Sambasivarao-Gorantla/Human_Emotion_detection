"""
Main training script - orchestrates the entire workflow
"""

import torch
from download_utils import download_dataset
from data_loader import DataProcessor
from model_builder import ModelBuilder
from trainer import TrainerBuilder
from metrics import MetricsComputer
import config


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print(" VISION TRANSFORMER FINE-TUNING PIPELINE")
    print("="*70 + "\n")
    
    # ===== STEP 1: Download Dataset =====
    print("[STEP 1] Downloading Dataset...")
    dataset_path = download_dataset(config.DATASET_NAME)
    
    # ===== STEP 2: Load and Prepare Data =====
    print("\n[STEP 2] Loading and Preparing Data...")
    data_processor = DataProcessor(dataset_path, config.PRETRAINED_MODEL)
    data_processor.load_dataset()
    
    # Get number of classes
    num_classes = data_processor.get_num_classes()
    class_names = data_processor.get_class_names()
    
    # Create dataloaders
    dataloaders = data_processor.get_dataloaders(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        batch_size=config.PER_DEVICE_BATCH_SIZE,
        shuffle=config.SHUFFLE
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    # ===== STEP 3: Build Model =====
    print("\n[STEP 3] Building Model...")
    model_builder = ModelBuilder(
        num_classes=num_classes,
        pretrained_model=config.PRETRAINED_MODEL,
        device=config.DEVICE
    )
    model = model_builder.load_model(freeze_backbone=config.FREEZE_BACKBONE)
    model_builder.print_model_info()
    
    # ===== STEP 4: Setup Training =====
    print("\n[STEP 4] Setting Up Training...")
    trainer_builder = TrainerBuilder(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=config.OUTPUT_DIR,
        learning_rate=config.LEARNING_RATE
    )
    
    training_args = trainer_builder.setup_training_args(
        num_epochs=config.NUM_EPOCHS,
        batch_size=config.PER_DEVICE_BATCH_SIZE,
        eval_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        save_total_limit=config.SAVE_TOTAL_LIMIT
    )
    
    trainer = trainer_builder.create_trainer(training_args)
    
    # ===== STEP 5: Train Model =====
    print("\n[STEP 5] Training Model...")
    print("="*70)
    trainer_builder.train()
    
    # ===== STEP 6: Evaluate on Validation Set =====
    print("\n[STEP 6] Evaluating on Validation Set...")
    val_results = trainer_builder.evaluate(val_loader.dataset)
    print(f"Validation Accuracy: {val_results.get('eval_accuracy', 0):.4f}")
    
    # ===== STEP 7: Evaluate on Test Set =====
    print("\n[STEP 7] Evaluating on Test Set...")
    test_results = trainer_builder.evaluate(test_loader.dataset)
    print(f"Test Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
    
    # ===== STEP 8: Get Predictions for Detailed Metrics =====
    print("\n[STEP 8] Computing Detailed Metrics...")
    predictions = trainer.predict(test_loader.dataset)
    test_predictions = predictions.predictions.argmax(axis=1)
    test_labels = predictions.label_ids
    
    detailed_metrics = MetricsComputer.compute_detailed_metrics(
        test_predictions, 
        test_labels, 
        class_names=class_names
    )
    
    MetricsComputer.print_metrics(detailed_metrics, class_names=class_names)
    
    # ===== STEP 9: Get Confusion Matrix =====
    print("\n[STEP 9] Generating Confusion Matrix...")
    cm = MetricsComputer.get_confusion_matrix(test_predictions, test_labels)
    print(f"\nConfusion Matrix:\n{cm}")
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
    return {
        'model': model,
        'trainer': trainer,
        'test_results': test_results,
        'detailed_metrics': detailed_metrics,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    results = main()
