import os
import torch

BASE_DIR = "/home/ab3379/work/data/MMEW_Final"
FEATURE_SAVE_DIR = os.path.join(BASE_DIR, "fused_features")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(FEATURE_SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Dataset Classes (6 core expressions)
CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# Dimensions
SEQUENCE_LENGTH = 16
VIT_DIM = 768       # Dv: ViT patch/sequence token dimension
MOTION_DIM = 256    # Dm: Motion / Optical Flow encoder dimension
FUSED_DIM = VIT_DIM + MOTION_DIM  # Dv + Dm

# Training Settings
BATCH_SIZE = 8
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"