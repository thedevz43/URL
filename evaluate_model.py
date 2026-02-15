"""
Comprehensive Model Evaluation Script
Evaluates the improved model on the full test dataset
"""

import numpy as np
import pickle
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import load_and_preprocess_data
from model import focal_loss

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix - Improved Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")

def evaluate_model():
    """Comprehensive model evaluation"""
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    print()
    
    # Load data
    print("[1] Loading test data...")
    DATA_PATH = 'data/malicious_phish.csv'
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, preprocessor = \
        load_and_preprocess_data(DATA_PATH, test_size=0.2, random_state=42, multi_input=True)
    
    print(f"✓ Test set size: {len(y_test)} samples")
    print()
    
    # Load model
    print("[2] Loading improved model...")
    model = keras.models.load_model('models/url_detector_improved.h5',
                                     custom_objects={'focal_loss_fixed': focal_loss()})
    print("✓ Model loaded")
    print()
    
    # Evaluate
    print("[3] Evaluating on test set...")
    print("-" * 80)
    
    # Get predictions
    y_pred_probs = model.predict(X_test, batch_size=256, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    eval_results = model.evaluate(X_test, y_test_cat, batch_size=256, verbose=0)
    test_loss = eval_results[0]
    test_accuracy = eval_results[1] if len(eval_results) > 1 else 0.0
    
    print()
    print("="*80)
    print("OVERALL METRICS")
    print("="*80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()
    
    # Classification report
    print("="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    class_names = list(preprocessor.label_encoder.classes_)
    print(classification_report(y_test, y_pred, 
                                target_names=class_names,
                                digits=4))
    
    # Confusion matrix
    print("="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    cm = confusion_matrix(y_test, y_pred)
    print("Rows = True labels, Columns = Predictions")
    print()
    
    # Print confusion matrix with labels
    col_width = max(len(name) for name in class_names) + 2
    print(" " * col_width, end="")
    for name in class_names:
        print(f"{name:>{col_width}}", end="")
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name:<{col_width}}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>{col_width}}", end="")
        print()
    print()
    
    # Calculate per-class accuracy
    print("="*80)
    print("PER-CLASS ACCURACY")
    print("="*80)
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i][i]
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{name:12} : {correct:6}/{total:6} = {accuracy:6.2f}%")
    print()
    
    # False positive analysis for benign class
    print("="*80)
    print("FALSE POSITIVE ANALYSIS (Benign misclassified)")
    print("="*80)
    benign_idx = list(class_names).index('benign')
    benign_mask = (y_test == benign_idx)
    benign_incorrect = benign_mask & (y_pred != benign_idx)
    
    if benign_incorrect.sum() > 0:
        print(f"Total benign samples: {benign_mask.sum()}")
        print(f"Misclassified as:")
        for i, name in enumerate(class_names):
            if i != benign_idx:
                count = ((y_pred[benign_incorrect] == i).sum())
                pct = count / benign_mask.sum() * 100
                print(f"  {name:12} : {count:6} ({pct:5.2f}%)")
    print()
    
    # Save confusion matrix
    print("[4] Saving visualizations...")
    plot_confusion_matrix(cm, class_names, 'models/evaluation_confusion_matrix.png')
    
    # Save metrics to file
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'classification_report': classification_report(y_test, y_pred, 
                                                       target_names=class_names,
                                                       output_dict=True),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    import json
    with open('models/evaluation_metrics_detailed.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ Metrics saved to models/evaluation_metrics_detailed.json")
    print()
    
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    evaluate_model()
