"""
Retrain Model with Augmented Data and Compare Performance

This script:
1. Loads augmented dataset
2. Retrains the multi-input CNN (same architecture)
3. Re-runs brand bias test
4. Re-runs calibration and stress tests
5. Compares old vs new performance

Goal: Reduce 82% brand false positive rate to <10%
"""

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from src.model import build_multi_input_cnn_model, focal_loss
from src.preprocess import load_and_preprocess_data
from src.adversarial_generators import BrandDomainGenerator
from src.calibration import ConfidenceCalibrator


def retrain_model_with_augmented_data(
    data_path: str = 'data/malicious_phish_augmented.csv',
    model_save_path: str = 'models/url_detector_augmented.h5',
    preprocessor_save_path: str = 'models/preprocessor_augmented.pkl',
    metadata_save_path: str = 'models/training_metadata_augmented.json',
    epochs: int = 10,
    batch_size: int = 256,
    patience: int = 3
):
    """
    Retrain model with augmented dataset
    
    Args:
        data_path: Path to augmented dataset
        model_save_path: Where to save trained model
        preprocessor_save_path: Where to save preprocessor
        metadata_save_path: Where to save training metadata
        epochs: Training epochs
        batch_size: Batch size
        patience: Early stopping patience
        
    Returns:
        Trained model, preprocessor, history, metadata
    """
    print("="*80)
    print("RETRAINING MODEL WITH AUGMENTED DATA")
    print("="*80)
    print()
    
    # Load and preprocess data
    print("[1/5] Loading and preprocessing augmented dataset...")
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, preprocessor = \
        load_and_preprocess_data(data_path)
    
    # Unpack multi-input data
    X_train_url, X_train_domain = X_train
    X_test_url, X_test_domain = X_test
    
    # Get class names from preprocessor
    class_names = list(preprocessor.label_encoder.classes_)
    n_classes = len(class_names)
    print(f"✓ Dataset loaded: {len(X_train_url)} train, {len(X_test_url)} test samples")
    print(f"✓ Classes: {class_names}")
    print()
    
    # Create model (same architecture as before)
    print("[2/5] Creating multi-input CNN model...")
    model = build_multi_input_cnn_model(
        vocab_size=preprocessor.vocab_size,
        max_url_length=preprocessor.max_url_length,
        max_domain_length=preprocessor.max_domain_length,
        num_classes=n_classes,
        embedding_dim=128,
        use_focal_loss=False  # We'll compile with focal loss manually
    )
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
    
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Check parameter constraint
    if total_params > 2_000_000:
        print(f"⚠️ Warning: Model has {total_params:,} parameters (exceeds 2M limit)")
    else:
        print(f"✓ Model meets parameter constraint (<2M)")
    print()
    
    # Compile model with class weights (same as original)
    print("[3/5] Compiling model...")
    
    # Calculate class weights
    y_train_labels = np.argmax(y_train_cat, axis=1)
    unique, counts = np.unique(y_train_labels, return_counts=True)
    total = len(y_train_labels)
    class_weights = {int(cls): total / (len(unique) * count) for cls, count in zip(unique, counts)}
    
    print("Class weights:")
    for cls_idx, weight in class_weights.items():
        print(f"  {class_names[cls_idx]:15} : {weight:.4f}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss(alpha=0.25, gamma=2.0),
        metrics=['accuracy']
    )
    print("✓ Model compiled with focal loss")
    print()
    
    # Train model
    print("[4/5] Training model...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Early stopping patience: {patience}")
    print()
    
    start_time = datetime.now()
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        [X_train_url, X_train_domain],
        y_train_cat,
        validation_data=([X_test_url, X_test_domain], y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    print()
    print(f"✓ Training completed in {training_time:.1f} seconds")
    print()
    
    # Evaluate model
    print("[5/5] Evaluating model...")
    test_loss, test_accuracy = model.evaluate(
        [X_test_url, X_test_domain],
        y_test_cat,
        verbose=0
    )
    
    print(f"✓ Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"✓ Test loss: {test_loss:.4f}")
    print()
    
    # Save model and preprocessor
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    
    print(f"Saving preprocessor to {preprocessor_save_path}...")
    with open(preprocessor_save_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_path': data_path,
        'model_path': model_save_path,
        'preprocessor_path': preprocessor_save_path,
        'architecture': 'multi_input_cnn',
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'training': {
            'epochs_run': len(history.history['loss']),
            'epochs_max': epochs,
            'batch_size': batch_size,
            'patience': patience,
            'training_time_seconds': training_time
        },
        'dataset': {
            'train_samples': int(len(X_train_url)),
            'test_samples': int(len(X_test_url)),
            'n_classes': n_classes,
            'class_names': class_names
        },
        'performance': {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1])
        },
        'class_weights': {class_names[k]: float(v) for k, v in class_weights.items()}
    }
    
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {metadata_save_path}")
    print()
    
    return model, preprocessor, history, metadata


def test_brand_bias(
    model,
    preprocessor,
    class_names,
    n_samples: int = 50
) -> dict:
    """
    Test for brand bias (false positive rate)
    
    Args:
        model: Trained model
        preprocessor: URL preprocessor
        class_names: List of class names
        n_samples: Number of brand URLs to test
        
    Returns:
        Test results dictionary
    """
    print("="*80)
    print("BRAND BIAS TEST")
    print("="*80)
    print()
    
    # Generate legitimate brand URLs
    print(f"Generating {n_samples} legitimate brand URLs...")
    brand_gen = BrandDomainGenerator()
    legitimate_urls = brand_gen.generate_legitimate_urls(n_samples=n_samples)
    
    print(f"✓ Generated URLs from top domains")
    print()
    
    # Test for false positives
    print("Testing for false positives...")
    
    urls = [item['url'] for item in legitimate_urls]
    
    # Encode URLs
    url_seqs = preprocessor.encode_urls(urls)
    domain_seqs = preprocessor.encode_domains(urls)
    
    # Predict
    predictions_probs = model.predict([url_seqs, domain_seqs], verbose=0)
    predictions = np.argmax(predictions_probs, axis=1)
    confidences = np.max(predictions_probs, axis=1)
    
    # Count false positives
    correct = 0
    false_positives = []
    
    for i, item in enumerate(legitimate_urls):
        pred_class = class_names[predictions[i]]
        confidence = confidences[i]
        
        if pred_class == 'benign':
            correct += 1
        else:
            false_positives.append({
                'domain': item['domain'],
                'url': item['url'],
                'predicted': pred_class,
                'confidence': float(confidence * 100)
            })
    
    # Calculate metrics
    fp_rate = (len(false_positives) / n_samples) * 100
    accuracy = (correct / n_samples) * 100
    
    # Print results
    print()
    print("-" * 80)
    print("Results:")
    print("-" * 80)
    print(f"Legitimate URLs tested: {n_samples}")
    print(f"Correctly classified as benign: {correct} ({accuracy:.2f}%)")
    print(f"False Positive Rate: {fp_rate:.2f}%")
    print()
    
    if false_positives:
        print(f"False Positives (first 10):")
        for i, fp in enumerate(false_positives[:10], 1):
            print(f"  {i:2}. {fp['domain']:30} → {fp['predicted']:12} ({fp['confidence']:.2f}%)")
        
        if len(false_positives) > 10:
            print(f"  ... and {len(false_positives) - 10} more")
    else:
        print("✓ No false positives detected!")
    
    print()
    
    # Verdict
    if fp_rate <= 5:
        print("✓ EXCELLENT: False positive rate ≤5%")
    elif fp_rate <= 10:
        print("✓ GOOD: False positive rate ≤10%")
    elif fp_rate <= 20:
        print("⚠️ ACCEPTABLE: False positive rate ≤20%")
    else:
        print("❌ HIGH: False positive rate >20%")
    
    print()
    
    return {
        'n_samples': n_samples,
        'correct': correct,
        'accuracy': accuracy,
        'fp_rate': fp_rate,
        'false_positives': false_positives
    }


def test_calibration(
    model,
    preprocessor,
    class_names,
    data_path: str = 'data/malicious_phish_augmented.csv',
    n_samples: int = 10000
) -> dict:
    """
    Test confidence calibration
    
    Args:
        model: Trained model
        preprocessor: URL preprocessor
        class_names: List of class names
        data_path: Path to dataset
        n_samples: Number of samples for calibration analysis
        
    Returns:
        Calibration results dictionary
    """
    print("="*80)
    print("CALIBRATION TEST")
    print("="*80)
    print()
    
    # Load test data
    print(f"Loading test data ({n_samples} samples)...")
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, _ = \
        load_and_preprocess_data(data_path)
    
    # Unpack multi-input data
    X_train_url, X_train_domain = X_train
    X_test_url, X_test_domain = X_test
    
    # Use subset for faster computation
    if len(X_test_url) > n_samples:
        indices = np.random.choice(len(X_test_url), n_samples, replace=False)
        X_test_url = X_test_url[indices]
        X_test_domain = X_test_domain[indices]
        y_test_cat = y_test_cat[indices]
    
    print(f"✓ Using {len(X_test_url)} test samples")
    print()
    
    # Predict
    print("Computing predictions...")
    y_pred_probs = model.predict([X_test_url, X_test_domain], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    
    # Compute calibration
    print("Computing calibration metrics...")
    calibrator = ConfidenceCalibrator()
    
    ece_results = calibrator.compute_ece(y_true, y_pred_probs, n_bins=15)
    conf_dist = calibrator.analyze_confidence_distribution(y_pred_probs)
    
    print()
    print("-" * 80)
    print("Results:")
    print("-" * 80)
    print(f"Expected Calibration Error (ECE): {ece_results['ECE']:.4f}")
    print(f"Maximum Calibration Error (MCE): {ece_results['MCE']:.4f}")
    print(f"Average Confidence: {conf_dist['mean_confidence']:.4f}")
    print()
    
    print("Confidence Distribution:")
    print(f"  High confidence (>90%): {conf_dist['high_confidence_rate']:.2%}")
    print(f"  Low confidence (<50%): {conf_dist['low_confidence_rate']:.2%}")
    print()
    
    # Verdict
    if ece_results['ECE'] < 0.05:
        print("✓ EXCELLENT: ECE < 0.05")
    elif ece_results['ECE'] < 0.10:
        print("✓ GOOD: ECE < 0.10")
    elif ece_results['ECE'] < 0.20:
        print("⚠️ ACCEPTABLE: ECE < 0.20")
    else:
        print("❌ POOR: ECE ≥ 0.20")
    
    print()
    
    return {
        'ECE': ece_results['ECE'],
        'MCE': ece_results['MCE'],
        'avg_confidence': conf_dist['mean_confidence'],
        'high_conf_rate': conf_dist['high_confidence_rate'],
        'low_conf_rate': conf_dist['low_confidence_rate']
    }


def compare_models(
    old_metadata_path: str = 'models/training_metadata_improved.json',
    new_metadata_path: str = 'models/training_metadata_augmented.json',
    old_brand_bias: dict = None,
    new_brand_bias: dict = None,
    old_calibration: dict = None,
    new_calibration: dict = None
):
    """
    Compare old and new model performance
    
    Args:
        old_metadata_path: Path to old model metadata
        new_metadata_path: Path to new model metadata
        old_brand_bias: Old brand bias test results
        new_brand_bias: New brand bias test results
        old_calibration: Old calibration results
        new_calibration: New calibration results
    """
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print()
    
    # Load metadata
    with open(old_metadata_path, 'r') as f:
        old_meta = json.load(f)
    
    with open(new_metadata_path, 'r') as f:
        new_meta = json.load(f)
    
    # Compare overall accuracy
    print("Overall Accuracy:")
    print(f"  Old model: {old_meta['performance']['test_accuracy']:.4f} ({old_meta['performance']['test_accuracy']*100:.2f}%)")
    print(f"  New model: {new_meta['performance']['test_accuracy']:.4f} ({new_meta['performance']['test_accuracy']*100:.2f}%)")
    
    acc_change = (new_meta['performance']['test_accuracy'] - old_meta['performance']['test_accuracy']) * 100
    if acc_change > 0:
        print(f"  Change: +{acc_change:.2f}% ✓")
    else:
        print(f"  Change: {acc_change:.2f}%")
    print()
    
    # Compare training set size
    print("Training Set Size:")
    print(f"  Old model: {old_meta['dataset']['train_samples']:,} samples")
    print(f"  New model: {new_meta['dataset']['train_samples']:,} samples")
    increase = new_meta['dataset']['train_samples'] - old_meta['dataset']['train_samples']
    print(f"  Increase: +{increase:,} samples ({100*increase/old_meta['dataset']['train_samples']:.1f}%)")
    print()
    
    # Compare brand bias (if available)
    if old_brand_bias and new_brand_bias:
        print("Brand Bias (False Positive Rate on Top Domains):")
        print(f"  Old model: {old_brand_bias['fp_rate']:.2f}% ❌")
        print(f"  New model: {new_brand_bias['fp_rate']:.2f}%", end='')
        
        if new_brand_bias['fp_rate'] <= 5:
            print(" ✓✓")
        elif new_brand_bias['fp_rate'] <= 10:
            print(" ✓")
        elif new_brand_bias['fp_rate'] < old_brand_bias['fp_rate']:
            print(" (improved)")
        else:
            print(" ❌")
        
        fp_improvement = old_brand_bias['fp_rate'] - new_brand_bias['fp_rate']
        print(f"  Improvement: {fp_improvement:.2f} percentage points")
        print(f"  Reduction: {100 * fp_improvement / old_brand_bias['fp_rate']:.1f}%")
        print()
    
    # Compare calibration (if available)
    if old_calibration and new_calibration:
        print("Confidence Calibration (ECE):")
        print(f"  Old model: {old_calibration['ECE']:.4f}")
        print(f"  New model: {new_calibration['ECE']:.4f}")
        
        ece_change = new_calibration['ECE'] - old_calibration['ECE']
        if ece_change < 0:
            print(f"  Change: {ece_change:.4f} (better) ✓")
        else:
            print(f"  Change: +{ece_change:.4f}")
        print()
    
    # Overall verdict
    print("="*80)
    print("VERDICT")
    print("="*80)
    print()
    
    if new_brand_bias and new_brand_bias['fp_rate'] <= 10:
        print("✅ SUCCESS: Brand false positive rate reduced to ≤10%")
        print("   Model is now suitable for production deployment")
    elif new_brand_bias and new_brand_bias['fp_rate'] < old_brand_bias['fp_rate'] * 0.5:
        print("✓ IMPROVED: Brand false positive rate reduced by >50%")
        print("  Further data augmentation may improve results")
    elif new_brand_bias:
        print("⚠️ PARTIAL IMPROVEMENT: Brand bias reduced but still high")
        print("   Consider additional data augmentation or domain whitelist")
    else:
        print("Model retrained successfully")
    
    print()


if __name__ == "__main__":
    """Main execution pipeline"""
    
    print("="*80)
    print("MODEL RETRAINING AND COMPARISON PIPELINE")
    print("="*80)
    print()
    print("This script will:")
    print("  1. Retrain model with augmented data")
    print("  2. Test brand bias (false positive rate)")
    print("  3. Test confidence calibration")
    print("  4. Compare with original model")
    print()
    input("Press Enter to continue...")
    print()
    
    # Step 1: Retrain model
    model, preprocessor, history, metadata = retrain_model_with_augmented_data(
        data_path='data/malicious_phish_augmented.csv',
        model_save_path='models/url_detector_augmented.h5',
        preprocessor_save_path='models/preprocessor_augmented.pkl',
        metadata_save_path='models/training_metadata_augmented.json',
        epochs=10,
        batch_size=256,
        patience=3
    )
    
    class_names = metadata['dataset']['class_names']
    
    # Step 2: Test brand bias
    new_brand_bias = test_brand_bias(
        model=model,
        preprocessor=preprocessor,
        class_names=class_names,
        n_samples=50
    )
    
    # Step 3: Test calibration
    new_calibration = test_calibration(
        model=model,
        preprocessor=preprocessor,
        class_names=class_names,
        data_path='data/malicious_phish_augmented.csv',
        n_samples=10000
    )
    
    # Step 4: Load old results for comparison
    # Load old brand bias from stress test report
    try:
        with open('models/stress_test_report.json', 'r') as f:
            stress_report = json.load(f)
            old_brand_bias = {
                'fp_rate': stress_report['tests']['brand_bias']['fp_rate'],
                'n_samples': stress_report['tests']['brand_bias']['n_samples']
            }
            old_calibration = {
                'ECE': stress_report['tests']['calibration']['ECE']
            }
    except:
        print("⚠️ Could not load old stress test results for comparison")
        old_brand_bias = None
        old_calibration = None
    
    # Step 5: Compare models
    compare_models(
        old_metadata_path='models/training_metadata_improved.json',
        new_metadata_path='models/training_metadata_augmented.json',
        old_brand_bias=old_brand_bias,
        new_brand_bias=new_brand_bias,
        old_calibration=old_calibration,
        new_calibration=new_calibration
    )
    
    # Save comparison results
    comparison_results = {
        'timestamp': datetime.now().isoformat(),
        'new_brand_bias': new_brand_bias,
        'new_calibration': new_calibration,
        'old_brand_bias': old_brand_bias,
        'old_calibration': old_calibration
    }
    
    with open('models/augmentation_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print("✓ Comparison results saved to models/augmentation_comparison.json")
    print()
    print("="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
