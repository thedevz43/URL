"""
Training script for the improved multi-input CNN model.

This script trains the enhanced architecture with:
- Multi-input processing (URL + Domain branches)
- Focal loss for better class imbalance handling
- Domain extraction for reduced false positives on legitimate brands

Usage:
    python train_improved_model.py
"""

import os
import sys
import numpy as np
from src.preprocess import load_and_preprocess_data, URLPreprocessor
from src.model import build_multi_input_cnn_model
from src.train import train_model, plot_training_history, save_training_metadata


def main():
    """Main training pipeline for improved model."""
    
    print("\n" + "=" * 80)
    print("IMPROVED MULTI-INPUT DNN FOR MALICIOUS URL DETECTION")
    print("=" * 80)
    print("\nEnhancements:")
    print("  ✓ Multi-input architecture (URL + Domain branches)")
    print("  ✓ Focal loss for class imbalance")
    print("  ✓ Domain extraction for brand recognition")
    print("  ✓ Reduced false positives on legitimate domains")
    print("=" * 80)
    
    # Configuration
    DATA_PATH = 'data/malicious_phish.csv'
    MODEL_SAVE_PATH = 'models/url_detector_improved.h5'
    PREPROCESSOR_SAVE_PATH = 'models/preprocessor_improved.pkl'
    HISTORY_PLOT_PATH = 'models/training_history_improved.png'
    METADATA_PATH = 'models/training_metadata_improved.json'
    
    EPOCHS = 30
    BATCH_SIZE = 128
    VALIDATION_SPLIT = 0.2
    USE_FOCAL_LOSS = True
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # ============================================================================
    # STEP 1: LOAD AND PREPROCESS DATA
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, preprocessor = \
        load_and_preprocess_data(
            DATA_PATH,
            test_size=VALIDATION_SPLIT,
            random_state=42,
            multi_input=True  # Enable multi-input mode
        )
    
    # X_train is now a list: [X_train_url, X_train_domain]
    X_train_url, X_train_domain = X_train
    X_test_url, X_test_domain = X_test
    
    print(f"\nData shapes:")
    print(f"  URL sequences: {X_train_url.shape} (train), {X_test_url.shape} (test)")
    print(f"  Domain sequences: {X_train_domain.shape} (train), {X_test_domain.shape} (test)")
    print(f"  Labels: {y_train_cat.shape} (train), {y_test_cat.shape} (test)")
    
    # Save preprocessor
    preprocessor.save(PREPROCESSOR_SAVE_PATH)
    print(f"\n✓ Preprocessor saved to {PREPROCESSOR_SAVE_PATH}")
    
    # ============================================================================
    # STEP 2: BUILD IMPROVED MODEL
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("STEP 2: MODEL ARCHITECTURE")
    print("=" * 80)
    
    model = build_multi_input_cnn_model(
        vocab_size=preprocessor.vocab_size,
        max_url_length=preprocessor.max_url_length,
        max_domain_length=preprocessor.max_domain_length,
        num_classes=preprocessor.num_classes,
        embedding_dim=128,
        use_focal_loss=USE_FOCAL_LOSS
    )
    
    # Display model summary
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    if total_params < 2_000_000:
        print(f"✓ Within 2M parameter budget ({total_params / 1_000_000:.2f}M)")
    else:
        print(f"⚠ Exceeds 2M parameter budget ({total_params / 1_000_000:.2f}M)")
    
    # ============================================================================
    # STEP 3: TRAIN MODEL
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("STEP 3: MODEL TRAINING")
    print("=" * 80)
    
    # Prepare multi-input data
    X_train_multi = [X_train_url, X_train_domain]
    X_test_multi = [X_test_url, X_test_domain]
    
    # Train
    history = train_model(
        model=model,
        X_train=X_train_multi,
        y_train_cat=y_train_cat,
        X_val=X_test_multi,
        y_val_cat=y_test_cat,
        y_train=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=MODEL_SAVE_PATH
    )
    
    # ============================================================================
    # STEP 4: SAVE TRAINING ARTIFACTS
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("STEP 4: SAVING TRAINING ARTIFACTS")
    print("=" * 80)
    
    # Plot training history
    plot_training_history(history, save_path=HISTORY_PLOT_PATH)
    print(f"✓ Training curves saved to {HISTORY_PLOT_PATH}")
    
    # Save metadata
    save_training_metadata(history, model, save_path=METADATA_PATH)
    print(f"✓ Training metadata saved to {METADATA_PATH}")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - IMPROVED MODEL")
    print("=" * 80)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    
    print(f"\nFinal Training Metrics:")
    print(f"  Accuracy: {final_train_acc:.4f}")
    print(f"  Loss: {final_train_loss:.4f}")
    
    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy: {final_val_acc:.4f}")
    print(f"  Loss: {final_val_loss:.4f}")
    
    print(f"\nBest Validation Metrics:")
    print(f"  Best Accuracy: {best_val_acc:.4f}")
    print(f"  Best Loss: {best_val_loss:.4f}")
    
    print(f"\nModel artifacts saved:")
    print(f"  • Model: {MODEL_SAVE_PATH}")
    print(f"  • Preprocessor: {PREPROCESSOR_SAVE_PATH}")
    print(f"  • Training history: {HISTORY_PLOT_PATH}")
    print(f"  • Metadata: {METADATA_PATH}")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Evaluate model: python evaluate_improved_model.py")
    print("  2. Test predictions: python test_improved_model.py")
    print("  3. Compare with original model performance")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
