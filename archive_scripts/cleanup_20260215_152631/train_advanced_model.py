"""
Train Advanced 3-Branch Model

Trains production-grade URL detector with:
- Branch A: URL character CNN
- Branch B: Domain character CNN
- Branch C: Handcrafted features
- MC Dropout for uncertainty
- Class-balanced focal loss
- Temperature scaling calibration
"""

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import sys

# Import custom modules
from src.preprocess import URLPreprocessor, load_and_preprocess_data
from src.advanced_model import build_advanced_3branch_model, focal_loss_with_label_smoothing
from src.feature_engineering import URLFeatureExtractor

print("="*80)
print("TRAINING ADVANCED 3-BRANCH URL DETECTOR")
print("="*80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_path': 'data/malicious_phish_final.csv',
    'model_save_path': 'models/url_detector_advanced.h5',
    'preprocessor_save_path': 'models/preprocessor_advanced.pkl',
    'feature_extractor_save_path': 'models/feature_extractor_advanced.pkl',
    'metadata_save_path': 'models/training_metadata_advanced.json',
    
    # Model architecture
    'vocab_size': 70,
    'max_url_length': 200,
    'max_domain_length': 100,
    'embedding_dim': 128,
    'use_rejection_class': False,  # Will add after initial training
    
    # Training hyperparameters (optimized for 676K dataset)
    'batch_size': 512,  # Larger batch for more stable gradients
    'epochs': 30,  # More epochs for large dataset
    'learning_rate': 0.0015,  # Slightly higher LR for faster convergence
    'validation_split': 0.12,  # More data for training
    'test_split': 0.12,  # More data for training
    'random_seed': 42,
    
    # Optimized class weights (based on final dataset: 676K URLs)
    # Inverse frequency scaling: weight = median_freq / class_freq
    'class_weights': {
        0: 0.35,  # benign (66.71% = 451,233) - reduced weight
        1: 1.65,  # defacement (14.09% = 95,308)
        2: 1.50,  # phishing (15.59% = 105,448)
        3: 6.50   # malware (3.61% = 24,395) - highest weight
    }
}

print("Configuration:")
for key, value in CONFIG.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for k, v in value.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {key}: {value}")
print()

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("[1/6] Loading and preprocessing data...")
print(f"      Dataset: {CONFIG['data_path']}")

# Load data
df = pd.read_csv(CONFIG['data_path'])
print(f"      Loaded: {len(df):,} URLs")

# Encode URLs and domains
preprocessor = URLPreprocessor(
    vocab_size=CONFIG['vocab_size'],
    max_url_length=CONFIG['max_url_length'],
    max_domain_length=CONFIG['max_domain_length']
)

urls = df['url'].values
labels = df['type'].values

print(f"      Encoding URLs and domains...")
url_sequences = preprocessor.fit_transform_urls(urls)
domain_sequences = preprocessor.transform_domains(urls)
encoded_labels = preprocessor.fit_transform_labels(labels)

print(f"      URL sequences: {url_sequences.shape}")
print(f"      Domain sequences: {domain_sequences.shape}")
print(f"      Labels: {encoded_labels.shape}")

# Extract handcrafted features
print(f"      Extracting handcrafted features...")
feature_extractor = URLFeatureExtractor()
features = feature_extractor.extract_batch(urls)
print(f"      Features: {features.shape} ({features.shape[1]} features per URL)")

# Stratified train/val/test split
from sklearn.model_selection import train_test_split

# First split: train+val vs test
X_url_temp, X_url_test, X_domain_temp, X_domain_test, X_feat_temp, X_feat_test, y_temp, y_test = train_test_split(
    url_sequences, domain_sequences, features, encoded_labels,
    test_size=CONFIG['test_split'],
    random_state=CONFIG['random_seed'],
    stratify=encoded_labels
)

# Second split: train vs val
X_url_train, X_url_val, X_domain_train, X_domain_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
    X_url_temp, X_domain_temp, X_feat_temp, y_temp,
    test_size=CONFIG['validation_split'] / (1 - CONFIG['test_split']),
    random_state=CONFIG['random_seed'],
    stratify=y_temp
)

print()
print(f"      Train set: {len(X_url_train):,} samples")
print(f"      Val set:   {len(X_url_val):,} samples")
print(f"      Test set:  {len(X_url_test):,} samples")
print()

# ============================================================================
# BUILD MODEL
# ============================================================================

print("[2/6] Building 3-branch architecture...")

model = build_advanced_3branch_model(
    vocab_size=CONFIG['vocab_size'],
    max_url_length=CONFIG['max_url_length'],
    max_domain_length=CONFIG['max_domain_length'],
    n_features=features.shape[1],
    n_classes=4,  # benign, defacement, phishing, malware
    embedding_dim=CONFIG['embedding_dim'],
    use_rejection=CONFIG['use_rejection_class']
)

# Display architecture
print()
model.summary()
print()

# Count parameters
total_params = model.count_params()
print(f"Total parameters: {total_params:,}")

if total_params > 2_000_000:
    print(f"⚠ WARNING: Model has {total_params:,} parameters (target: <2M)")
else:
    print(f"✓ Model within parameter budget (<2M)")
print()

# ============================================================================
# COMPILE MODEL
# ============================================================================

print("[3/6] Compiling model...")

# Use focal loss with label smoothing
loss_function = focal_loss_with_label_smoothing(
    gamma=2.0,
    alpha=0.25,
    label_smoothing=0.1
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss=loss_function,
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print(f"      Optimizer: Adam (lr={CONFIG['learning_rate']})")
print(f"      Loss: Focal loss with label smoothing")
print(f"      Metrics: Accuracy, Precision, Recall")
print()

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("[4/6] Training model...")
print()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        CONFIG['model_save_path'],
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# Train
history = model.fit(
    [X_url_train, X_domain_train, X_feat_train],
    y_train,
    validation_data=([X_url_val, X_domain_val, X_feat_val], y_val),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    class_weight=CONFIG['class_weights'],
    callbacks=callbacks,
    verbose=1
)

print()
print("✓ Training complete!")
print()

# ============================================================================
# EVALUATE MODEL
# ============================================================================

print("[5/6] Evaluating on test set...")

test_loss, test_acc, test_precision, test_recall = model.evaluate(
    [X_url_test, X_domain_test, X_feat_test],
    y_test,
    batch_size=CONFIG['batch_size'],
    verbose=0
)

print(f"      Test Loss:      {test_loss:.4f}")
print(f"      Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"      Test Precision: {test_precision:.4f}")
print(f"      Test Recall:    {test_recall:.4f}")
print()

# Detailed evaluation
print("      Computing detailed metrics...")
y_pred_probs = model.predict(
    [X_url_test, X_domain_test, X_feat_test],
    batch_size=CONFIG['batch_size'],
    verbose=0
)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report, confusion_matrix

print()
print("Classification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=['benign', 'defacement', 'phishing', 'malware'],
    digits=4
))

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)
print()

# ============================================================================
# SAVE ARTIFACTS
# ============================================================================

print("[6/6] Saving model and artifacts...")

# Save preprocessor
with open(CONFIG['preprocessor_save_path'], 'wb') as f:
    pickle.dump(preprocessor, f)
print(f"      ✓ Saved preprocessor: {CONFIG['preprocessor_save_path']}")

# Save feature extractor
with open(CONFIG['feature_extractor_save_path'], 'wb') as f:
    pickle.dump(feature_extractor, f)
print(f"      ✓ Saved feature extractor: {CONFIG['feature_extractor_save_path']}")

# Save model (already saved by ModelCheckpoint, but save final version too)
model.save(CONFIG['model_save_path'].replace('.h5', '_final.h5'))
print(f"      ✓ Saved final model: {CONFIG['model_save_path'].replace('.h5', '_final.h5')}")

# Save training metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'config': CONFIG,
    'dataset_stats': {
        'total_samples': len(df),
        'train_samples': len(X_url_train),
        'val_samples': len(X_url_val),
        'test_samples': len(X_url_test),
        'class_distribution': {
            label: int(count) 
            for label, count in df['type'].value_counts().items()
        }
    },
    'model_stats': {
        'total_parameters': int(total_params),
        'architecture': '3-branch (URL CNN + Domain CNN + Features)',
        'branches': {
            'url_cnn': 'Conv1D blocks + MC Dropout + GlobalMaxPool',
            'domain_cnn': 'Conv1D blocks + MC Dropout + GlobalMaxPool',
            'features': 'Dense layers + handcrafted features'
        }
    },
    'training_results': {
        'final_train_loss': float(history.history['loss'][-1]),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'epochs_trained': len(history.history['loss'])
    },
    'test_results': {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall)
    },
    'classification_report': classification_report(
        y_true, y_pred,
        target_names=['benign', 'defacement', 'phishing', 'malware'],
        output_dict=True
    )
}

with open(CONFIG['metadata_save_path'], 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"      ✓ Saved metadata: {CONFIG['metadata_save_path']}")

print()
print("="*80)
print("TRAINING COMPLETE!")
print("="*80)
print()
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print(f"Model: {CONFIG['model_save_path']}")
print(f"Parameters: {total_params:,}")
print()
print("Next steps:")
print("  1. Run comprehensive evaluation (test_suite.py)")
print("  2. Apply temperature scaling calibration")
print("  3. Test on brand bias, international domains, adversarial attacks")
print("  4. Set up drift monitoring in production")
print()
