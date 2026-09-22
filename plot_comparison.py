import os
import matplotlib.pyplot as plt
import numpy as np

# Experiment Metrics Comparison
classes = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Experiment 1: Unstratified Split (87% Acc)
exp1_f1 = [0.00, 0.88, 0.00, 0.89, 0.86, 0.94]
exp1_recall = [0.00, 0.93, 0.00, 1.00, 0.75, 0.94]

# Setup bar positions
x = np.arange(len(classes))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: F1-Score Breakdown
axes[0].bar(x - width/2, exp1_f1, width, label='Unstratified (87% Acc)', color='#3498db')
axes[0].set_title('Per-Class F1-Score Baseline', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(classes)
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel('F1-Score')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
axes[0].legend()

# Panel 2: Recall Breakdown
axes[1].bar(x - width/2, exp1_recall, width, label='Unstratified (87% Acc)', color='#e74c3c')
axes[1].set_title('Per-Class Recall (Sensitivity to Rare Classes)', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel('Recall')
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()

# Save to distinct comparison file
save_path = "experiment_comparison_summary.png"
plt.savefig(save_path, dpi=300)
print(f"Comparison plot saved as '{save_path}'")