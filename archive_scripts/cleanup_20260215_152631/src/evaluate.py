"""
Evaluation module for malicious URL detection model.

Provides comprehensive evaluation metrics, confusion matrix,
and detailed analysis of model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import json


def evaluate_model(model, X_test, y_test, y_test_cat, preprocessor, 
                   save_path='../models/evaluation_results.png'):
    """
    Comprehensive evaluation of the trained model.
    
    Generates:
    1. Confusion matrix
    2. Classification report (precision, recall, F1-score per class)
    3. Overall metrics
    4. Visualizations
    5. Analysis of difficult cases
    
    Args:
        model: Trained Keras model
        X_test (np.ndarray): Test sequences
        y_test (np.ndarray): Test labels (integer encoded)
        y_test_cat (np.ndarray): Test labels (one-hot encoded)
        preprocessor: URLPreprocessor instance for label decoding
        save_path (str): Path to save evaluation visualizations
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 80)
    
    # Make predictions
    print("\nGenerating predictions...")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Get class names
    class_names = preprocessor.label_encoder.classes_
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics (macro average treats all classes equally)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Weighted average (accounts for class imbalance)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Print overall metrics
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nMacro Average (treats all classes equally):")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall:    {recall_macro:.4f}")
    print(f"  F1-Score:  {f1_macro:.4f}")
    print("\nWeighted Average (accounts for class imbalance):")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall:    {recall_weighted:.4f}")
    print(f"  F1-Score:  {f1_weighted:.4f}")
    
    # Confusion matrix
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    cm = confusion_matrix(y_test, y_pred)
    print("\n", cm)
    
    # Classification report
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    print("\n", classification_report(
        y_test, 
        y_pred, 
        target_names=class_names,
        digits=4,
        zero_division=0
    ))
    
    # Per-class analysis
    print("\n" + "=" * 80)
    print("PER-CLASS ANALYSIS")
    print("=" * 80)
    
    for i, class_name in enumerate(class_names):
        class_total = np.sum(y_test == i)
        class_correct = np.sum((y_test == i) & (y_pred == i))
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        
        print(f"\n{class_name.upper()}:")
        print(f"  Total samples: {class_total}")
        print(f"  Correctly classified: {class_correct}")
        print(f"  Class accuracy: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
        
        # Find misclassifications
        misclassified_as = {}
        for j, other_class in enumerate(class_names):
            if i != j:
                count = np.sum((y_test == i) & (y_pred == j))
                if count > 0:
                    misclassified_as[other_class] = count
        
        if misclassified_as:
            print(f"  Misclassified as:")
            for other_class, count in sorted(misclassified_as.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / class_total) * 100
                print(f"    - {other_class}: {count} ({percentage:.2f}%)")
    
    # Analyze difficult cases
    print("\n" + "=" * 80)
    print("ANALYSIS: WHICH ATTACKS ARE HARDEST TO DETECT?")
    print("=" * 80)
    
    analyze_difficult_cases(cm, class_names)
    
    # Create visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_evaluation_results(cm, class_names, y_test, y_pred, y_pred_prob, save_path)
    
    # Prepare metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {}
    }
    
    for i, class_name in enumerate(class_names):
        class_total = int(np.sum(y_test == i))
        class_correct = int(np.sum((y_test == i) & (y_pred == i)))
        metrics['per_class_metrics'][class_name] = {
            'total': class_total,
            'correct': class_correct,
            'accuracy': float(class_correct / class_total) if class_total > 0 else 0.0
        }
    
    # Save metrics
    metrics_path = save_path.replace('.png', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    return metrics


def analyze_difficult_cases(cm, class_names):
    """
    Analyze which attack types are most difficult to detect.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): List of class names
    """
    print("\nDETAILED ANALYSIS:\n")
    
    # Calculate per-class recall (true positive rate)
    recalls = []
    for i in range(len(class_names)):
        total = cm[i, :].sum()
        correct = cm[i, i]
        recall = correct / total if total > 0 else 0
        recalls.append((class_names[i], recall, total))
    
    # Sort by recall (ascending)
    recalls.sort(key=lambda x: x[1])
    
    print("Classes ranked by detection difficulty (lowest recall = hardest):\n")
    for rank, (class_name, recall, total) in enumerate(recalls, 1):
        print(f"{rank}. {class_name.upper()}")
        print(f"   Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"   Total samples: {total}")
        
        if recall < 0.80:
            print(f"   ⚠️  LOW RECALL - Many {class_name} URLs are being missed!")
        elif recall < 0.90:
            print(f"   ⚡ MODERATE - Some {class_name} URLs are missed.")
        else:
            print(f"   ✓ GOOD - Most {class_name} URLs are correctly detected.")
        print()
    
    print("\nWHY ARE SOME ATTACKS HARDER TO DETECT?\n")
    print("1. PHISHING URLs:")
    print("   - Often mimic legitimate domains (e.g., 'paypa1.com' vs 'paypal.com')")
    print("   - Small character differences are hard for character-level models")
    print("   - May use legitimate hosting services\n")
    
    print("2. MALWARE URLs:")
    print("   - Can be hosted on compromised legitimate sites")
    print("   - File extensions may not always be obvious")
    print("   - Often use obfuscation techniques\n")
    
    print("3. DEFACEMENT URLs:")
    print("   - Target legitimate websites")
    print("   - URL structure may look normal")
    print("   - Only the content is changed, not the URL pattern\n")
    
    print("4. BENIGN URLs:")
    print("   - Easy to detect due to large training samples")
    print("   - Diverse patterns from legitimate sources")
    print("   - But unusual legitimate URLs may be misclassified\n")


def plot_evaluation_results(cm, class_names, y_test, y_pred, y_pred_prob, save_path):
    """
    Create comprehensive evaluation visualizations.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): List of class names
        y_test (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_prob (np.ndarray): Prediction probabilities
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Model Evaluation - Malicious URL Detector', fontsize=16, fontweight='bold')
    
    # Plot 1: Confusion Matrix (counts)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0, 0],
        cbar_kws={'label': 'Count'}
    )
    axes[0, 0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # Plot 2: Confusion Matrix (normalized)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='RdYlGn',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0, 1],
        cbar_kws={'label': 'Percentage'},
        vmin=0,
        vmax=1
    )
    axes[0, 1].set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')
    
    # Plot 3: Per-class accuracy
    class_accuracies = []
    for i in range(len(class_names)):
        total = cm[i, :].sum()
        correct = cm[i, i]
        accuracy = correct / total if total > 0 else 0
        class_accuracies.append(accuracy)
    
    bars = axes[1, 0].bar(class_names, class_accuracies, color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'])
    axes[1, 0].set_title('Per-Class Recall', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].axhline(y=0.9, color='r', linestyle='--', label='90% threshold', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{acc:.2%}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    # Plot 4: Prediction confidence distribution
    max_probs = np.max(y_pred_prob, axis=1)
    correct_mask = (y_test == y_pred)
    
    axes[1, 1].hist(
        max_probs[correct_mask],
        bins=50,
        alpha=0.7,
        label='Correct predictions',
        color='green',
        edgecolor='black'
    )
    axes[1, 1].hist(
        max_probs[~correct_mask],
        bins=50,
        alpha=0.7,
        label='Incorrect predictions',
        color='red',
        edgecolor='black'
    )
    axes[1, 1].set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Maximum Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nEvaluation visualizations saved to {save_path}")


if __name__ == "__main__":
    print("\nThis module should be imported and used with trained model and test data.")
    print("See main.py for complete evaluation pipeline.")
