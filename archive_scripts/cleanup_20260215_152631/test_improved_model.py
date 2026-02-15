"""
Testing script for the improved multi-input CNN model.

This script demonstrates inference with the enhanced model and
compares predictions on URLs that were previously misclassified.

Usage:
    python test_improved_model.py
"""

import numpy as np
import tensorflow as tf
from src.preprocess import URLPreprocessor


def predict_url(url, model, preprocessor):
    """
    Predict malicious URL class for a single URL.
    
    Args:
        url (str): URL to classify
        model: Trained Keras model
        preprocessor: URLPreprocessor instance
        
    Returns:
        tuple: (predicted_class, class_probabilities)
    """
    # Encode URL
    url_seq = preprocessor.encode_urls([url])
    
    # Encode domain
    domain_seq = preprocessor.encode_domains([url])
    
    # Predict
    probs = model.predict([url_seq, domain_seq], verbose=0)[0]
    
    # Get predicted class
    pred_class_idx = np.argmax(probs)
    pred_class = preprocessor.decode_label(pred_class_idx)
    
    return pred_class, probs


def print_prediction(url, pred_class, probs, preprocessor):
    """Pretty print prediction results."""
    
    class_names = preprocessor.label_encoder.classes_
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULT")
    print("=" * 80)
    print(f"\nURL: {url}")
    print(f"\nPredicted Class: {pred_class.upper()}")
    print(f"Confidence: {probs.max():.4f} ({probs.max() * 100:.2f}%)")
    
    print(f"\nAll class probabilities:")
    for class_name, prob in zip(class_names, probs):
        bar_length = int(prob * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {class_name:12} [{bar}] {prob:.4f} ({prob * 100:.2f}%)")
    
    # Risk assessment
    confidence = probs.max()
    if confidence > 0.9:
        if pred_class == 'benign':
            risk = "✓ LOW RISK - URL appears safe"
        else:
            risk = f"⚠️  HIGH RISK - URL classified as {pred_class.upper()}"
    elif confidence > 0.7:
        risk = f"⚠️  MODERATE RISK - Possible {pred_class.upper()}"
    else:
        risk = "❓ UNCERTAIN - Low confidence prediction"
    
    print(f"\nRisk Assessment:")
    print(f"  {risk}")
    print("=" * 80)


def main():
    """Main testing function."""
    
    print("\n" + "=" * 80)
    print("IMPROVED MODEL TESTING - MULTI-INPUT CNN")
    print("=" * 80)
    
    # Load model and preprocessor
    print("\n[1] Loading improved model...")
    MODEL_PATH = 'models/url_detector_improved.h5'
    PREPROCESSOR_PATH = 'models/preprocessor_improved.pkl'
    
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'focal_loss_fixed': lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(y_true, y_pred)}
        )
        print(f"✓ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease train the improved model first:")
        print("  python train_improved_model.py")
        return
    
    preprocessor = URLPreprocessor.load(PREPROCESSOR_PATH)
    print(f"✓ Preprocessor loaded from {PREPROCESSOR_PATH}")
    
    # ============================================================================
    # TEST CASES: Previously misclassified URLs
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("TESTING ON PREVIOUSLY MISCLASSIFIED URLS")
    print("=" * 80)
    
    test_urls = [
        # Legitimate sites (previously flagged as phishing)
        ("https://github.com/user/repo", "benign", "Legitimate GitHub URL"),
        ("https://www.amazon.com/product/B08N5WRWNW", "benign", "Legitimate Amazon product"),
        ("https://docs.python.org/3/library/index.html", "benign", "Python documentation"),
        ("https://www.microsoft.com", "benign", "Microsoft official site"),
        ("https://stackoverflow.com/questions/12345", "benign", "Stack Overflow question"),
        
        # URL shorteners (ambiguous)
        ("http://bit.ly/3x9k2lm", "phishing", "URL shortener"),
        
        # Clear malicious patterns
        ("http://example.com/download.exe", "malware", "Malware download"),
        ("http://bankofamerica-verify.tk/account/update.php", "phishing", "Phishing bank URL"),
        ("http://free-iphone-giveaway.com/claim?ref=123abc", "phishing", "Phishing giveaway"),
        ("http://192.168.1.1/admin/login.php", "phishing", "IP-based login page"),
        ("http://totallylegit.com/virus.exe?download=true", "malware", "Obvious malware"),
    ]
    
    print("\nTesting improved model on challenge URLs...")
    print("Focus: Legitimate brand domains that were previously misclassified\n")
    
    results = []
    
    for url, expected_class, description in test_urls:
        pred_class, probs = predict_url(url, model, preprocessor)
        
        # Check if correct
        correct = "✓" if pred_class == expected_class else "✗"
        confidence = probs.max()
        
        results.append({
            'url': url,
            'description': description,
            'expected': expected_class,
            'predicted': pred_class,
            'confidence': confidence,
            'correct': correct == "✓"
        })
        
        print(f"{correct} {description}")
        print(f"  URL: {url[:70]}...")
        print(f"  Expected: {expected_class.upper()} | Predicted: {pred_class.upper()} ({confidence:.2%})")
        print()
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total
    
    print(f"\nOverall Performance:")
    print(f"  Total tests: {total}")
    print(f"  Correct: {correct}")
    print(f"  Incorrect: {total - correct}")
    print(f"  Accuracy: {accuracy:.2%}")
    
    # Focus on legitimate sites
    legit_results = [r for r in results if r['expected'] == 'benign']
    legit_correct = sum(1 for r in legit_results if r['correct'])
    
    print(f"\nLegitimate Site Detection (KEY IMPROVEMENT):")
    print(f"  Total legitimate sites: {len(legit_results)}")
    print(f"  Correctly identified: {legit_correct}")
    print(f"  False positives: {len(legit_results) - legit_correct}")
    print(f"  Accuracy: {legit_correct / len(legit_results):.2%}")
    
    if legit_correct / len(legit_results) > 0.8:
        print("\n  ✓ SIGNIFICANT IMPROVEMENT on legitimate domains!")
    elif legit_correct / len(legit_results) > 0.6:
        print("\n  ⚠ MODERATE IMPROVEMENT on legitimate domains")
    else:
        print("\n  ✗ NEEDS FURTHER TUNING for legitimate domains")
    
    # Malicious detection
    malicious_results = [r for r in results if r['expected'] != 'benign']
    malicious_correct = sum(1 for r in malicious_results if r['correct'])
    
    print(f"\nMalicious URL Detection:")
    print(f"  Total malicious URLs: {len(malicious_results)}")
    print(f"  Correctly identified: {malicious_correct}")
    print(f"  False negatives: {len(malicious_results) - malicious_correct}")
    print(f"  Accuracy: {malicious_correct / len(malicious_results):.2%}")
    
    print("\n" + "=" * 80)
    
    # Detailed view option
    print("\nFor detailed predictions, run:")
    print("  python test_improved_model.py --detailed")
    print("\nTo test custom URL:")
    print("  python test_improved_model.py --url <your_url>")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--url' and len(sys.argv) > 2:
            # Test single URL
            url = sys.argv[2]
            
            print("\n" + "=" * 80)
            print("SINGLE URL PREDICTION")
            print("=" * 80)
            
            try:
                model = tf.keras.models.load_model(
                    'models/url_detector_improved.h5',
                    custom_objects={'focal_loss_fixed': lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(y_true, y_pred)}
                )
                preprocessor = URLPreprocessor.load('models/preprocessor_improved.pkl')
                
                pred_class, probs = predict_url(url, model, preprocessor)
                print_prediction(url, pred_class, probs, preprocessor)
                
            except Exception as e:
                print(f"\nError: {e}")
                print("\nPlease train the improved model first:")
                print("  python train_improved_model.py")
        else:
            print("\nUsage:")
            print("  python test_improved_model.py                  # Run test suite")
            print("  python test_improved_model.py --url <url>      # Test single URL")
    else:
        main()
