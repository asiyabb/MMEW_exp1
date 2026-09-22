import os
import model
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import config
from dataset import MicroMacroDataset
from model import MicroToMacroGuidedTransformer

def evaluate_guided_pipeline():
    excel_path = os.path.join(config.BASE_DIR, "MMEW_Micro_Exp.xlsx")
    image_dir = os.path.join(config.BASE_DIR, "Micro_Expression")
    
    dataset = MicroMacroDataset(excel_path=excel_path, image_dir=image_dir)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    
    # Use exact same seed split as training for evaluation consistency
    torch.manual_seed(42)
    _, val_ds = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    device = torch.device(config.DEVICE)
    model = MicroToMacroGuidedTransformer().to(device)
    
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_guided_model_87acc.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
    # Update line 35 in evaluate.py:
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)    
    print(f"Loaded checkpoint from: {ckpt_path}\n")
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for frames, flows, labels in val_loader:
            frames, flows = frames.to(device), flows.to(device)
            _, macro_logits = model(frames, flows)
            preds = macro_logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Class Indices (0 to 5)
    class_indices = list(range(len(config.CLASSES)))
    
    # Compute Classification Metrics (labels=class_indices explicitly handles missing validation classes)
    print("--- EVALUATION REPORT (Guided Macro Head) ---")
    print(classification_report(
        all_labels, 
        all_preds, 
        labels=class_indices, 
        target_names=config.CLASSES, 
        zero_division=0
    ))
    
    cm = confusion_matrix(all_labels, all_preds, labels=class_indices)
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-6)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=class_indices, zero_division=0
    )
    
    df_metrics = pd.DataFrame({
        'Class': config.CLASSES,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Plot Evaluation Figures
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Normalized Confusion Matrix
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Purples",
                xticklabels=config.CLASSES, yticklabels=config.CLASSES, ax=axes[0], cbar=False)
    axes[0].set_title("Guided Macro Head: Normalized Confusion Matrix", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # Panel 2: Per-Class Precision, Recall, and F1-Score
    df_melted = pd.melt(df_metrics, id_vars=['Class'], value_vars=['Precision', 'Recall', 'F1-Score'], 
                        var_name='Metric', value_name='Score')
    sns.barplot(data=df_melted, x='Class', y='Score', hue='Metric', ax=axes[1], palette='magma')
    axes[1].set_title("Per-Class Performance Metrics", fontsize=12, fontweight='bold')
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    save_path = os.path.join(config.BASE_DIR, "guided_experiment_results.png")
    plt.savefig(save_path, dpi=300)
    print(f"Results chart saved to '{save_path}'")

if __name__ == "__main__":
    evaluate_guided_pipeline()