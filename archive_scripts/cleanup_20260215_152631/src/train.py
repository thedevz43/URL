"""
Training module for malicious URL detection model.

Handles model training with class weights, validation, and history logging.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.utils import class_weight
import json
from datetime import datetime


def calculate_class_weights(y_train):
    """
    Calculate class weights to handle imbalanced dataset.
    
    Why class weights?
    - Dataset is heavily imbalanced (benign: 65%, others: much less)
    - Without weights, model will bias toward majority class
    - Class weights penalize misclassification of minority classes more
    
    Args:
        y_train (np.ndarray): Integer-encoded training labels
        
    Returns:
        dict: Class weights dictionary {class_id: weight}
    """
    print("\n" + "=" * 80)
    print("CALCULATING CLASS WEIGHTS")
    print("=" * 80)
    
    # Compute class weights using sklearn
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Convert to dictionary
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("\nClass weights (to handle imbalance):")
    for class_id, weight in class_weight_dict.items():
        print(f"  Class {class_id}: {weight:.4f}")
    
    print("\nHigher weights mean the model will focus more on that class during training.")
    
    return class_weight_dict


def create_callbacks(model_path, patience=5):
    """
    Create training callbacks for model checkpointing and early stopping.
    
    Args:
        model_path (str): Path to save best model
        patience (int): Number of epochs to wait before early stopping
        
    Returns:
        list: List of Keras callbacks
    """
    callbacks = [
        # Save best model based on validation loss
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        
        # Early stopping if validation loss doesn't improve
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True,
            mode='min'
        ),
        
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7,
            mode='min'
        )
    ]
    
    print(f"\nCallbacks configured:")
    print(f"  - ModelCheckpoint: Save best model to {model_path}")
    print(f"  - EarlyStopping: Patience = {patience}")
    print(f"  - ReduceLROnPlateau: Factor = 0.5, Patience = 3")
    
    return callbacks


def train_model(model, X_train, y_train_cat, X_val, y_val_cat, 
                y_train, epochs=20, batch_size=128, model_save_path='../models/url_detector.h5'):
    """
    Train the malicious URL detection model.
    
    Supports both single-input and multi-input models:
    - Single-input: X_train is np.ndarray
    - Multi-input: X_train is list of [url_sequences, domain_sequences]
    
    Training strategy:
    1. Use class weights to handle imbalance
    2. Monitor validation loss for overfitting
    3. Save best model checkpoint
    4. Early stopping if no improvement
    5. Reduce learning rate on plateau
    
    Args:
        model: Compiled Keras model
        X_train: Training sequences (np.ndarray or list for multi-input)
        y_train_cat (np.ndarray): Training labels (one-hot)
        X_val: Validation sequences (np.ndarray or list for multi-input)
        y_val_cat (np.ndarray): Validation labels (one-hot)
        y_train (np.ndarray): Training labels (integer) for class weights
        epochs (int): Maximum number of epochs
        batch_size (int): Batch size for training
        model_save_path (str): Path to save trained model
        
    Returns:
        history: Training history object
    """
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    
    # Create callbacks
    callbacks = create_callbacks(model_save_path, patience=5)
    
    # Display training configuration
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Handle multi-input vs single-input
    if isinstance(X_train, list):
        # Multi-input model
        print(f"  Model type: Multi-input (URL + Domain)")
        print(f"  Training samples: {len(X_train[0])}")
        print(f"  Validation samples: {len(X_val[0])}")
        print(f"  Validation split: {len(X_val[0]) / (len(X_train[0]) + len(X_val[0])) * 100:.1f}%")
    else:
        # Single-input model
        print(f"  Model type: Single-input (URL only)")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Validation split: {len(X_val) / (len(X_train) + len(X_val)) * 100:.1f}%")
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    return history


def plot_training_history(history, save_path='../models/training_history.png'):
    """
    Plot and save training history.
    
    Visualizes:
    - Loss curves (train vs validation)
    - Accuracy curves (train vs validation)
    - Helps identify overfitting or underfitting
    
    Args:
        history: Keras training history object
        save_path (str): Path to save the plot
    """
    print(f"\nGenerating training history plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History - Malicious URL Detector', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    
    # Print final metrics
    print(f"\nFinal training metrics:")
    print(f"  Train Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Val Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  Train Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")


def save_training_metadata(history, model, save_path='../models/training_metadata.json'):
    """
    Save training metadata for reproducibility and analysis.
    
    Args:
        history: Keras training history object
        model: Trained Keras model
        save_path (str): Path to save metadata JSON
    """
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'model_parameters': int(model.count_params()),
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTraining metadata saved to {save_path}")


if __name__ == "__main__":
    print("\nThis module should be imported and used with actual training data.")
    print("See main.py for complete training pipeline.")
