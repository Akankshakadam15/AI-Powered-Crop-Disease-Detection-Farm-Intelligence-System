import torch
import matplotlib.pyplot as plt
import os

def save_checkpoint(model, optimizer, epoch, best_val_acc, path):
    """Save training checkpoint to resume later."""
    torch.save({
        'epoch'          : epoch,
        'model_state'    : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_acc'   : best_val_acc,
    }, path)
    print(f"  💾 Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, path):
    """Load checkpoint to resume training."""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch  = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f"✅ Resuming from Epoch {start_epoch} | Best Val Acc: {best_val_acc:.2f}%")
        return start_epoch, best_val_acc
    else:
        print("No checkpoint found → Starting fresh!")
        return 1, 0.0


def plot_results(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot and save training curves."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses,   label='Val Loss',   color='orange')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', color='blue')
    plt.plot(val_accs,   label='Val Acc',   color='orange')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📊 Plot saved: {save_path}")

    plt.show()