"""
Test individual modules before running the full pipeline.
Useful for debugging and understanding each component.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_preprocessing():
    """Test the preprocessing module."""
    print("\n" + "="*80)
    print("TESTING PREPROCESSING MODULE")
    print("="*80)
    
    from src.preprocess import URLPreprocessor
    
    # Initialize preprocessor
    preprocessor = URLPreprocessor(max_length=200)
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "http://malicious-site.com/phish?id=123",
        "www.example.com/path/to/page.html"
    ]
    
    print("\nTesting URL encoding:")
    for url in test_urls:
        sequence = preprocessor.url_to_sequence(url)
        print(f"\nURL: {url}")
        print(f"Length: {len(url)} characters")
        print(f"Encoded (first 20): {sequence[:20]}")
    
    # Test batch encoding
    encoded = preprocessor.encode_urls(test_urls)
    print(f"\nBatch encoded shape: {encoded.shape}")
    
    print("\n✓ Preprocessing module working correctly")


def test_model():
    """Test the model building."""
    print("\n" + "="*80)
    print("TESTING MODEL MODULE")
    print("="*80)
    
    from src.model import build_char_cnn_model
    
    # Build model with dummy parameters
    vocab_size = 100
    max_length = 200
    num_classes = 4
    
    model = build_char_cnn_model(vocab_size, max_length, num_classes)
    
    print("\nModel Summary:")
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    print("\n✓ Model module working correctly")


def test_full_preprocessing():
    """Test full preprocessing pipeline with actual data."""
    print("\n" + "="*80)
    print("TESTING FULL PREPROCESSING PIPELINE (SMALL SAMPLE)")
    print("="*80)
    
    from src.preprocess import load_and_preprocess_data
    import pandas as pd
    
    # Load small sample of data
    print("\nLoading small sample (10,000 URLs) for testing...")
    df = pd.read_csv('data/malicious_phish.csv', nrows=10000)
    
    # Save temporary sample
    df.to_csv('data/sample_data.csv', index=False)
    
    # Test preprocessing
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, preprocessor = \
        load_and_preprocess_data('data/sample_data.csv', test_size=0.2)
    
    print(f"\nSample preprocessing complete:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train_cat shape: {y_train_cat.shape}")
    print(f"  Vocabulary size: {preprocessor.vocab_size}")
    
    # Clean up
    os.remove('data/sample_data.csv')
    
    print("\n✓ Full preprocessing pipeline working correctly")


def test_inference():
    """Test inference if model is trained."""
    print("\n" + "="*80)
    print("TESTING INFERENCE (requires trained model)")
    print("="*80)
    
    if not os.path.exists('models/url_detector.h5'):
        print("\n⚠️  Model not found. Train the model first:")
        print("   python main.py --train")
        return
    
    from tensorflow import keras
    from src.preprocess import URLPreprocessor
    import numpy as np
    
    # Load model and preprocessor
    print("\nLoading model and preprocessor...")
    model = keras.models.load_model('models/url_detector.h5')
    preprocessor = URLPreprocessor.load('models/preprocessor.pkl')
    
    # Test URL
    test_url = "https://www.google.com"
    print(f"\nTesting URL: {test_url}")
    
    # Preprocess
    url_encoded = preprocessor.encode_urls([test_url])
    
    # Predict
    prediction = model.predict(url_encoded, verbose=0)[0]
    predicted_class = preprocessor.label_encoder.classes_[np.argmax(prediction)]
    confidence = prediction[np.argmax(prediction)]
    
    print(f"\nPredicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("\nAll probabilities:")
    for i, label in enumerate(preprocessor.label_encoder.classes_):
        print(f"  {label}: {prediction[i]:.4f}")
    
    print("\n✓ Inference working correctly")


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test individual modules')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--preprocess', action='store_true', help='Test preprocessing')
    parser.add_argument('--model', action='store_true', help='Test model building')
    parser.add_argument('--full', action='store_true', help='Test full preprocessing pipeline')
    parser.add_argument('--inference', action='store_true', help='Test inference')
    
    args = parser.parse_args()
    
    if args.all or not any([args.preprocess, args.model, args.full, args.inference]):
        test_preprocessing()
        test_model()
        test_full_preprocessing()
        test_inference()
    else:
        if args.preprocess:
            test_preprocessing()
        if args.model:
            test_model()
        if args.full:
            test_full_preprocessing()
        if args.inference:
            test_inference()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
