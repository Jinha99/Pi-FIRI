# plot.py

import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_metrics(history, save_path='training_logs/loss_plot.png'):
    sns.set(style='whitegrid')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Total loss
    if 'loss' in history and 'val_loss' in history:
        axs[0].plot(history['loss'], label='loss', color='tab:blue')
        axs[0].plot(history['val_loss'], label='val_loss', color='tab:orange')
        axs[0].set_title("Total Loss")
        axs[0].legend(frameon=False)
        axs[0].grid(True)

    # 2. Classification loss
    if 'cls_loss' in history and 'val_cls_loss' in history:
        axs[1].plot(history['cls_loss'], label='cls_loss', color='tab:green')
        axs[1].plot(history['val_cls_loss'], label='val_cls_loss', color='tab:red')
        axs[1].set_title("Classification Loss")
        axs[1].legend(frameon=False)
        axs[1].grid(True)

    # 3. Box regression loss
    if 'box_loss' in history and 'val_box_loss' in history:
        axs[2].plot(history['box_loss'], label='box_loss', color='tab:purple')
        axs[2].plot(history['val_box_loss'], label='val_box_loss', color='tab:brown')
        axs[2].set_title("Box Regression Loss")
        axs[2].legend(frameon=False)
        axs[2].grid(True)

    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
