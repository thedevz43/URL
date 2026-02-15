"""
Main script for Malicious URL Detection using Deep Neural Networks.

This script orchestrates the complete end-to-end pipeline:
1. Data loading and preprocessing
2. Model building
3. Training with class weights
4. Evaluation with comprehensive metrics
5. Model saving and loading
6. Real-time inference on new URLs

Author: Senior ML Engineer & Cybersecurity Researcher
Date: 2026
"""

import os
import sys
import numpy as np
from tensorflow import keras

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import load_and_preprocess_data, URLPreprocessor
from src.model import build_char_cnn_model
from src.train import train_model, plot_training_history, save_training_metadata
from src.evaluate import evaluate_model


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def train_pipeline(data_path='data/malicious_phish.csv', epochs=20, batch_size=128):
    """
    Complete training pipeline.
    
    Args:
        data_path (str): Path to dataset CSV
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
    """
    print_banner("MALICIOUS URL DETECTION - TRAINING PIPELINE")
    
    # Step 1: Load and preprocess data
    print_banner("STEP 1: DATA LOADING & PREPROCESSING")
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, preprocessor = \
        load_and_preprocess_data(data_path, test_size=0.2, random_state=42)
    
    # Split training data for validation
    val_split = 0.15
    split_idx = int(len(X_train) * (1 - val_split))
    
    X_train_final = X_train[:split_idx]
    y_train_final_cat = y_train_cat[:split_idx]
    y_train_final = y_train[:split_idx]
    
    X_val = X_train[split_idx:]
    y_val_cat = y_train_cat[split_idx:]
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train_final)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Step 2: Build model
    print_banner("STEP 2: MODEL ARCHITECTURE")
    model = build_char_cnn_model(
        vocab_size=preprocessor.vocab_size,
        max_length=preprocessor.max_length,
        num_classes=preprocessor.num_classes,
        embedding_dim=128
    )
    
    # Display model summary
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)
    model.summary()
    
    # Step 3: Train model
    print_banner("STEP 3: MODEL TRAINING")
    history = train_model(
        model=model,
        X_train=X_train_final,
        y_train_cat=y_train_final_cat,
        X_val=X_val,
        y_val_cat=y_val_cat,
        y_train=y_train_final,
        epochs=epochs,
        batch_size=batch_size,
        model_save_path='models/url_detector.h5'
    )
    
    # Step 4: Plot training history
    print_banner("STEP 4: TRAINING HISTORY VISUALIZATION")
    plot_training_history(history, save_path='models/training_history.png')
    
    # Step 5: Save metadata
    save_training_metadata(history, model, save_path='models/training_metadata.json')
    
    # Step 6: Save preprocessor
    print_banner("STEP 5: SAVING ARTIFACTS")
    preprocessor.save('models/preprocessor.pkl')
    print("Model artifacts saved:")
    print("  - models/url_detector.h5")
    print("  - models/preprocessor.pkl")
    print("  - models/training_history.png")
    print("  - models/training_metadata.json")
    
    # Step 7: Evaluate model
    print_banner("STEP 6: MODEL EVALUATION")
    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        y_test_cat=y_test_cat,
        preprocessor=preprocessor,
        save_path='models/evaluation_results.png'
    )
    
    print_banner("TRAINING PIPELINE COMPLETE")
    print("All artifacts saved to models/ directory")
    print("\nNext steps:")
    print("  1. Review evaluation_results.png for model performance")
    print("  2. Run inference on new URLs using predict_url() function")
    print("  3. Deploy model for production use")
    
    return model, preprocessor, metrics


def load_trained_model(model_path='models/url_detector.h5', 
                       preprocessor_path='models/preprocessor.pkl'):
    """
    Load a trained model and preprocessor from disk.
    
    Args:
        model_path (str): Path to saved model
        preprocessor_path (str): Path to saved preprocessor
        
    Returns:
        tuple: (model, preprocessor)
    """
    print_banner("LOADING TRAINED MODEL")
    
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    
    print(f"\nLoading preprocessor from {preprocessor_path}...")
    preprocessor = URLPreprocessor.load(preprocessor_path)
    print("✓ Preprocessor loaded successfully")
    
    return model, preprocessor


def predict_url(url, model, preprocessor, verbose=True):
    """
    Predict the class of a single URL.
    
    Args:
        url (str): URL string to classify
        model: Trained Keras model
        preprocessor: URLPreprocessor instance
        verbose (bool): Print detailed output
        
    Returns:
        dict: Prediction results with class and probabilities
    """
    # Preprocess URL
    url_sequence = preprocessor.encode_urls([url])
    
    # Make prediction
    pred_prob = model.predict(url_sequence, verbose=0)[0]
    pred_class_idx = np.argmax(pred_prob)
    pred_class = preprocessor.decode_label(pred_class_idx)
    confidence = pred_prob[pred_class_idx]
    
    if verbose:
        print("\n" + "=" * 80)
        print("URL CLASSIFICATION RESULT")
        print("=" * 80)
        print(f"\nURL: {url}")
        print(f"\nPredicted Class: {pred_class.upper()}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        print(f"\nAll class probabilities:")
        for i, class_name in enumerate(preprocessor.label_encoder.classes_):
            prob = pred_prob[i]
            bar_length = int(prob * 50)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {class_name:12s} [{bar}] {prob:.4f} ({prob*100:.2f}%)")
        
        # Risk assessment
        print(f"\nRisk Assessment:")
        if pred_class == 'benign':
            if confidence > 0.9:
                print("  ✓ LOW RISK - URL appears safe")
            else:
                print("  ⚠️  MODERATE - Further inspection recommended")
        else:
            if confidence > 0.8:
                print(f"  ⚠️  HIGH RISK - URL classified as {pred_class.upper()}")
                print(f"  Action: Block or flag for review")
            else:
                print(f"  ⚠️  MODERATE RISK - Possible {pred_class.upper()}")
                print(f"  Action: Additional verification needed")
    
    return {
        'url': url,
        'predicted_class': pred_class,
        'confidence': float(confidence),
        'probabilities': {
            class_name: float(pred_prob[i]) 
            for i, class_name in enumerate(preprocessor.label_encoder.classes_)
        }
    }


def inference_demo(model, preprocessor):
    """
    Interactive demo for URL classification.
    
    Args:
        model: Trained Keras model
        preprocessor: URLPreprocessor instance
    """
    print_banner("INFERENCE DEMO - MALICIOUS URL DETECTOR")
    
    # Test URLs (mix of benign and malicious)
    test_urls = [
        "https://www.google.com",
        "http://www.amazon.com/shop",
        "br-icloud.com.br",  # Phishing example from dataset
        "signin.eby.de.zukruygxctzmmqi.civpro.co.za",  # Phishing
        "http://www.824555.com/app/member/SportOption.php?uid=guest&langx=gb",  # Malware
        "http://www.github.com/repos",
        "https://www.paypal.com",
        "https://docs.google.com/spreadsheet/viewform?formkey=dGg2Z1lCUHlSdjllTVNRUW50TFIzSkE6MQ"  # Phishing
    ]
    
    print("Testing model on sample URLs...\n")
    
    results = []
    for i, url in enumerate(test_urls, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_urls)}")
        result = predict_url(url, model, preprocessor, verbose=True)
        results.append(result)
    
    print_banner("INFERENCE DEMO COMPLETE")
    
    # Summary
    print("\nSummary of predictions:")
    print(f"{'URL':<50} {'Predicted Class':<15} {'Confidence':<10}")
    print("-" * 80)
    for result in results:
        url_display = result['url'][:47] + '...' if len(result['url']) > 50 else result['url']
        print(f"{url_display:<50} {result['predicted_class']:<15} {result['confidence']:<10.2%}")


def interactive_mode(model, preprocessor):
    """
    Interactive mode for testing custom URLs.
    
    Args:
        model: Trained Keras model
        preprocessor: URLPreprocessor instance
    """
    print_banner("INTERACTIVE MODE - ENTER URLs FOR CLASSIFICATION")
    print("Type 'exit' or 'quit' to stop\n")
    
    while True:
        try:
            url = input("Enter URL to classify: ").strip()
            
            if url.lower() in ['exit', 'quit', '']:
                print("\nExiting interactive mode...")
                break
            
            predict_url(url, model, preprocessor, verbose=True)
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a valid URL\n")


def main():
    """
    Main entry point for the application.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Malicious URL Detection using Deep Neural Networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py --train
  
  # Train with custom epochs and batch size
  python main.py --train --epochs 30 --batch-size 256
  
  # Run inference demo on sample URLs
  python main.py --demo
  
  # Interactive mode to test custom URLs
  python main.py --interactive
  
  # Predict a single URL
  python main.py --predict "https://suspicious-site.com/login"
        """
    )
    
    parser.add_argument('--train', action='store_true',
                       help='Train a new model')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size (default: 128)')
    parser.add_argument('--demo', action='store_true',
                       help='Run inference demo on sample URLs')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode for testing custom URLs')
    parser.add_argument('--predict', type=str,
                       help='Predict class for a single URL')
    
    args = parser.parse_args()
    
    # Train mode
    if args.train:
        model, preprocessor, metrics = train_pipeline(
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Run demo after training
        print("\nWould you like to run the inference demo? (y/n): ", end='')
        response = input().strip().lower()
        if response == 'y':
            inference_demo(model, preprocessor)
    
    # Demo mode
    elif args.demo:
        model, preprocessor = load_trained_model()
        inference_demo(model, preprocessor)
    
    # Interactive mode
    elif args.interactive:
        model, preprocessor = load_trained_model()
        interactive_mode(model, preprocessor)
    
    # Single URL prediction
    elif args.predict:
        model, preprocessor = load_trained_model()
        predict_url(args.predict, model, preprocessor, verbose=True)
    
    # Default: show help
    else:
        parser.print_help()
        print("\n" + "=" * 80)
        print("QUICK START")
        print("=" * 80)
        print("\n1. First time? Train the model:")
        print("   python main.py --train")
        print("\n2. Test on sample URLs:")
        print("   python main.py --demo")
        print("\n3. Test your own URLs:")
        print("   python main.py --interactive")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
