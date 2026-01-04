"""
Configuration file for Vision Transformer fine-tuning project
"""

# Dataset configuration
DATASET_NAME = "samithsachidanandan/human-face-emotions"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Model configuration
PRETRAINED_MODEL = "Falconsai/nsfw_image_detection"
FREEZE_BACKBONE = True  # Freeze ViT layers, train only classifier

# Training configuration
OUTPUT_DIR = "./results"
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
PER_DEVICE_BATCH_SIZE = 1024
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 2
LOAD_BEST_MODEL = True
METRIC_FOR_BEST_MODEL = "accuracy"
GREATER_IS_BETTER = True

# Device configuration
DEVICE = "cuda"  # or "cpu"

# Data loading
SHUFFLE = True
NUM_WORKERS = 4
