"""
Generate Training & Validation Loss Curves
High-quality JPG visualization for documentation
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

print("Loading training metadata...")

# Load metadata with training history
with open("models/production/metadata.json", "r") as f:
    metadata = json.load(f)

print("Generating Training & Validation Loss Curves...")

# Create figure with high resolution
fig, ax = plt.subplots(figsize=(12, 7))

# Extract data
epochs = range(1, len(metadata['history']['loss']) + 1)
train_loss = metadata['history']['loss']
val_loss = metadata['history']['val_loss']

# Plot training loss
ax.plot(epochs, train_loss, 'b-', linewidth=2.5, label='Training Loss', marker='o', 
        markersize=4, markevery=2, alpha=0.8)

# Plot validation loss
ax.plot(epochs, val_loss, 'r-', linewidth=2.5, label='Validation Loss', marker='s', 
        markersize=4, markevery=2, alpha=0.8)

# Mark best model epoch (epoch 27)
best_epoch = 27
best_val_loss = metadata['best_val_loss']
ax.plot(best_epoch, best_val_loss, 'g*', markersize=20, label=f'Best Model (Epoch {best_epoch})',
        zorder=5)

# Add vertical line at best epoch
ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.5)

# Annotations
ax.annotate(f'Best: {best_val_loss:.6f}', 
            xy=(best_epoch, best_val_loss), 
            xytext=(best_epoch + 2, best_val_loss + 0.002),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green'))

# Final values annotation
final_train = train_loss[-1]
final_val = val_loss[-1]
ax.text(len(epochs) - 1, final_train + 0.001, f'{final_train:.5f}', 
        fontsize=9, ha='right', va='bottom', color='blue', fontweight='bold')
ax.text(len(epochs) - 1, final_val + 0.001, f'{final_val:.5f}', 
        fontsize=9, ha='right', va='bottom', color='red', fontweight='bold')

# Styling
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss (Categorical Crossentropy)', fontsize=14, fontweight='bold')
ax.set_title('Training & Validation Loss Curves\nv7 Production Model - 30 Epochs', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Set limits for better visualization
ax.set_xlim([0, len(epochs) + 1])
ax.set_ylim([0, max(max(train_loss), max(val_loss)) * 1.1])

# Add statistics box
stats_text = f"""Training Statistics:
Initial Loss: {train_loss[0]:.5f}
Final Loss: {train_loss[-1]:.5f}
Reduction: {((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.1f}%

Validation Statistics:
Initial Loss: {val_loss[0]:.5f}
Final Loss: {val_loss[-1]:.5f}
Best Loss: {best_val_loss:.6f}"""

ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace')

plt.tight_layout()

# Save high-quality JPG
output_path = output_dir / "training_validation_loss_curves.jpg"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n{'='*60}")
print("✅ Loss curves generated successfully!")
print(f"📁 Saved to: {output_path.absolute()}")
print(f"{'='*60}")
print("\nVisualization Details:")
print(f"  • Total Epochs: {len(epochs)}")
print(f"  • Best Epoch: {best_epoch}")
print(f"  • Best Val Loss: {best_val_loss:.6f}")
print(f"  • Final Train Loss: {train_loss[-1]:.5f}")
print(f"  • Final Val Loss: {val_loss[-1]:.5f}")
print(f"  • Loss Reduction: {((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.1f}%")
print(f"\n✓ High-resolution JPG (300 DPI) ready for documentation")
