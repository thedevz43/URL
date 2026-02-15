"""
Generate visualization graphs for the comprehensive report
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create visualizations directory
viz_dir = Path("visualizations")
viz_dir.mkdir(exist_ok=True)

print("Loading data from JSON files...")

# Load metadata
with open("models/production/metadata.json", "r") as f:
    metadata = json.load(f)

# Load evaluation metrics
with open("models/production/evaluation_metrics.json", "r") as f:
    eval_metrics = json.load(f)

# Load performance report
with open("models/production/performance_report.json", "r") as f:
    perf_report = json.load(f)

print("Generating visualizations...")

# 1. Training History - Loss Curves
print("1. Creating training history (loss curves)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, len(metadata['history']['loss']) + 1)
train_loss = metadata['history']['loss']
val_loss = metadata['history']['val_loss']

ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Model Loss Over Training', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Accuracy curves
train_acc = metadata['history']['accuracy']
val_acc = metadata['history']['val_accuracy']

ax2.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy')
ax2.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Model Accuracy Over Training', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.9, 1.0])

plt.tight_layout()
plt.savefig(viz_dir / "training_history.jpg", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: training_history.jpg")

# 2. Confusion Matrix Heatmap
print("2. Creating confusion matrix heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))

cm = np.array(eval_metrics['confusion_matrix'])
classes = ['Benign', 'Defacement', 'Malware', 'Phishing']

# Normalize confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=classes, yticklabels=classes, 
            cbar_kws={'label': 'Percentage'}, ax=ax,
            linewidths=0.5, linecolor='gray')

ax.set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Actual Class', fontsize=12)
ax.set_xlabel('Predicted Class', fontsize=12)

# Add counts as text
for i in range(len(classes)):
    for j in range(len(classes)):
        text = ax.text(j + 0.5, i + 0.7, f'n={cm[i, j]:,}',
                      ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig(viz_dir / "confusion_matrix.jpg", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: confusion_matrix.jpg")

# 3. Per-Class Performance Metrics
print("3. Creating per-class performance comparison...")
fig, ax = plt.subplots(figsize=(12, 6))

classes = list(eval_metrics['classification_report'].keys())[:-3]  # Exclude avg rows
metrics = ['precision', 'recall', 'f1-score']
colors = ['#3498db', '#e74c3c', '#2ecc71']

x = np.arange(len(classes))
width = 0.25

for idx, metric in enumerate(metrics):
    values = [eval_metrics['classification_report'][cls][metric] * 100 for cls in classes]
    ax.bar(x + idx * width, values, width, label=metric.capitalize(), color=colors[idx], alpha=0.8)

ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([c.capitalize() for c in classes])
ax.legend(fontsize=10)
ax.set_ylim([90, 100])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for idx, metric in enumerate(metrics):
    values = [eval_metrics['classification_report'][cls][metric] * 100 for cls in classes]
    for i, v in enumerate(values):
        ax.text(i + idx * width, v + 0.3, f'{v:.1f}%', 
               ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(viz_dir / "per_class_performance.jpg", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: per_class_performance.jpg")

# 4. v7 Enhancement Performance
print("4. Creating v7 enhancement metrics...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# False Positive Rate
fp_rate = perf_report['performance_metrics']['false_positive_rate'] * 100
ax1.bar(['v7 System'], [fp_rate], color='#e74c3c', alpha=0.7, width=0.5)
ax1.axhline(y=5, color='orange', linestyle='--', linewidth=2, label='Target (5%)')
ax1.set_ylabel('False Positive Rate (%)', fontsize=11)
ax1.set_title('False Positive Rate', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 10])
ax1.legend()
ax1.text(0, fp_rate + 0.3, f'{fp_rate:.1f}%', ha='center', fontsize=12, fontweight='bold')

# Detection Rate
det_rate = perf_report['performance_metrics']['malicious_detection_rate'] * 100
ax2.bar(['v7 System'], [det_rate], color='#2ecc71', alpha=0.7, width=0.5)
ax2.axhline(y=95, color='orange', linestyle='--', linewidth=2, label='Target (95%)')
ax2.set_ylabel('Detection Rate (%)', fontsize=11)
ax2.set_title('Malicious Detection Rate', fontsize=12, fontweight='bold')
ax2.set_ylim([90, 101])
ax2.legend()
ax2.text(0, det_rate - 2, f'{det_rate:.0f}%', ha='center', fontsize=12, fontweight='bold')

# Inference Time
inf_time = perf_report['performance_metrics']['average_inference_time_ms']
ax3.bar(['v7 System'], [inf_time], color='#3498db', alpha=0.7, width=0.5)
ax3.axhline(y=100, color='orange', linestyle='--', linewidth=2, label='Target (100ms)')
ax3.set_ylabel('Inference Time (ms)', fontsize=11)
ax3.set_title('Average Inference Time', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 120])
ax3.legend()
ax3.text(0, inf_time + 3, f'{inf_time}ms', ha='center', fontsize=12, fontweight='bold')

# 4-Tier Thresholds Visualization
tiers = ['Tier 1\n(Always Block)', 'Tier 2A\n(Rep-Based)', 'Tier 2B\n(Elite Only)', 'Tier 3\n(Allow)']
thresholds = [
    perf_report['enhancement_system']['tier_1_threshold'],
    perf_report['enhancement_system']['tier_2a_threshold'],
    perf_report['enhancement_system']['tier_2b_threshold'],
    0
]
colors_tier = ['#c0392b', '#e67e22', '#f39c12', '#27ae60']

ax4.barh(tiers, thresholds, color=colors_tier, alpha=0.7)
ax4.set_xlabel('Confidence Threshold', fontsize=11)
ax4.set_title('4-Tier Decision Thresholds', fontsize=12, fontweight='bold')
ax4.set_xlim([0, 1])

for i, v in enumerate(thresholds):
    if v > 0:
        ax4.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(viz_dir / "v7_enhancement_metrics.jpg", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: v7_enhancement_metrics.jpg")

# 5. Test Results Summary Dashboard
print("5. Creating test results summary dashboard...")
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Overall Accuracy
ax1 = fig.add_subplot(gs[0, 0])
test_acc = eval_metrics['test_accuracy'] * 100
ax1.pie([test_acc, 100-test_acc], labels=['Correct', 'Incorrect'], 
        colors=['#2ecc71', '#ecf0f1'], autopct='%1.2f%%',
        startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title(f'Test Accuracy\n{test_acc:.2f}%', fontsize=12, fontweight='bold')

# Class Distribution
ax2 = fig.add_subplot(gs[0, 1])
class_counts = [eval_metrics['classification_report'][cls]['support'] for cls in ['benign', 'defacement', 'malware', 'phishing']]
class_names = ['Benign', 'Defacement', 'Malware', 'Phishing']
colors_pie = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12']
ax2.pie(class_counts, labels=class_names, colors=colors_pie, autopct='%1.1f%%',
        textprops={'fontsize': 9})
ax2.set_title('Test Set Distribution', fontsize=12, fontweight='bold')

# Model Parameters
ax3 = fig.add_subplot(gs[0, 2])
ax3.text(0.5, 0.7, f"{metadata['model_parameters']:,}", ha='center', va='center',
         fontsize=24, fontweight='bold', color='#3498db')
ax3.text(0.5, 0.4, 'Total Parameters', ha='center', va='center',
         fontsize=12, color='#7f8c8d')
ax3.text(0.5, 0.2, f"Model Size: 13.3 MB", ha='center', va='center',
         fontsize=10, color='#95a5a6')
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.axis('off')

# F1-Scores Comparison
ax4 = fig.add_subplot(gs[1, :])
f1_scores = [eval_metrics['classification_report'][cls]['f1-score'] * 100 for cls in ['benign', 'defacement', 'malware', 'phishing']]
bars = ax4.barh(class_names, f1_scores, color=colors_pie, alpha=0.8)
ax4.set_xlabel('F1-Score (%)', fontsize=11)
ax4.set_title('F1-Score by Class', fontsize=12, fontweight='bold')
ax4.set_xlim([90, 100])
ax4.grid(True, alpha=0.3, axis='x')

for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    ax4.text(score - 1.5, i, f'{score:.2f}%', va='center', ha='right',
            fontsize=10, fontweight='bold', color='white')

plt.suptitle('Test Results Summary Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig(viz_dir / "test_results_dashboard.jpg", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: test_results_dashboard.jpg")

# 6. Training Progress Summary
print("6. Creating training progress summary...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Loss improvement
ax1.bar(['Initial\n(Epoch 1)', 'Final\n(Epoch 30)'], 
        [train_loss[0], train_loss[-1]], 
        color=['#e74c3c', '#2ecc71'], alpha=0.7)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training Loss: Start vs End', fontsize=12, fontweight='bold')
ax1.text(0, train_loss[0] + 0.001, f'{train_loss[0]:.4f}', ha='center', fontsize=10)
ax1.text(1, train_loss[-1] + 0.001, f'{train_loss[-1]:.4f}', ha='center', fontsize=10)

# Accuracy improvement
ax2.bar(['Initial\n(Epoch 1)', 'Final\n(Epoch 30)'], 
        [train_acc[0] * 100, train_acc[-1] * 100], 
        color=['#e74c3c', '#2ecc71'], alpha=0.7)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_title('Training Accuracy: Start vs End', fontsize=12, fontweight='bold')
ax2.set_ylim([90, 100])
ax2.text(0, train_acc[0] * 100 + 0.3, f'{train_acc[0]*100:.2f}%', ha='center', fontsize=10)
ax2.text(1, train_acc[-1] * 100 + 0.3, f'{train_acc[-1]*100:.2f}%', ha='center', fontsize=10)

# Best model info
best_epoch = 27  # From training logs - best validation accuracy
ax3.text(0.5, 0.7, f"Epoch {best_epoch}", ha='center', va='center',
         fontsize=20, fontweight='bold', color='#2ecc71')
ax3.text(0.5, 0.5, 'Best Model Selected', ha='center', va='center',
         fontsize=12, color='#7f8c8d')
ax3.text(0.5, 0.35, f"Val Accuracy: {metadata['best_val_accuracy']*100:.2f}%", 
         ha='center', va='center', fontsize=11, color='#34495e')
ax3.text(0.5, 0.22, f"Val Loss: {metadata['best_val_loss']:.6f}", 
         ha='center', va='center', fontsize=11, color='#34495e')
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.axis('off')

# Training duration
epochs_data = list(range(1, 31))
milestone_epochs = [1, 5, 10, 15, 20, 25, 30]
milestone_acc = [train_acc[e-1] * 100 for e in milestone_epochs]

ax4.plot(milestone_epochs, milestone_acc, 'o-', linewidth=2, 
         markersize=8, color='#3498db', label='Train Accuracy')
ax4.set_xlabel('Epoch', fontsize=11)
ax4.set_ylabel('Accuracy (%)', fontsize=11)
ax4.set_title('Training Progress (Key Epochs)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([92, 99])

for e, acc in zip(milestone_epochs, milestone_acc):
    ax4.text(e, acc + 0.2, f'{acc:.1f}%', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(viz_dir / "training_progress_summary.jpg", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: training_progress_summary.jpg")

print("\n" + "="*60)
print("✅ All visualizations generated successfully!")
print(f"📁 Saved in: {viz_dir.absolute()}")
print("="*60)
print("\nGenerated files:")
print("  1. training_history.jpg - Loss and accuracy curves")
print("  2. confusion_matrix.jpg - Normalized confusion matrix")
print("  3. per_class_performance.jpg - Precision, recall, F1 comparison")
print("  4. v7_enhancement_metrics.jpg - v7 system performance")
print("  5. test_results_dashboard.jpg - Overall results summary")
print("  6. training_progress_summary.jpg - Training progression")
