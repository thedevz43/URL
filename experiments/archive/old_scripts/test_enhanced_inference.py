"""
Test Enhanced Inference System

Compare raw model predictions vs enhanced predictions
on brand domains to measure FP reduction
"""

import numpy as np
import pickle
from keras.models import load_model
from enhanced_inference import EnhancedPredictor
from domain_reputation import get_reputation_scorer


def load_brand_domains():
    """Load 50 legitimate brand domains for testing"""
    return [
        'https://google.com/search',
        'https://facebook.com/profile',
        'https://amazon.com/products',
        'https://microsoft.com/windows',
        'https://apple.com/iphone',
        'https://youtube.com/watch',
        'https://twitter.com/home',
        'https://instagram.com/explore',
        'https://linkedin.com/jobs',
        'https://netflix.com/browse',
        'https://github.com/repositories',
        'https://stackoverflow.com/questions',
        'https://reddit.com/r/all',
        'https://wikipedia.org/wiki',
        'https://zoom.us/join',
        'https://paypal.com/myaccount',
        'https://ebay.com/deals',
        'https://salesforce.com/login',
        'https://oracle.com/cloud',
        'https://adobe.com/products',
        'https://spotify.com/browse',
        'https://dropbox.com/home',
        'https://slack.com/workspace',
        'https://discord.com/channels',
        'https://twitch.tv/directory',
        'https://pinterest.com/ideas',
        'https://outlook.com/mail',
        'https://gmail.com/inbox',
        'https://wordpress.com/start',
        'https://medium.com/topics',
        'https://chase.com/login',
        'https://bankofamerica.com/account',
        'https://wellsfargo.com/signin',
        'https://capitalone.com/mycard',
        'https://discover.com/credit-cards',
        'https://bestbuy.com/cart',
        'https://target.com/products',
        'https://walmart.com/shop',
        'https://homedepot.com/store',
        'https://ikea.com/products',
        'https://nike.com/new',
        'https://espn.com/scores',
        'https://nba.com/games',
        'https://unity.com/products',
        'https://steam.com/store',
        'https://airbnb.com/explore',
        'https://booking.com/hotels',
        'https://coursera.org/courses',
        'https://khanacademy.org/learn',
        'https://duolingo.com/practice',
    ]


def load_malicious_urls():
    """Load sample malicious URLs for detection rate testing"""
    return [
        'http://g00gle.com/signin',  # Typosquatting
        'http://paypal-secure.tk/login',  # Suspicious TLD
        'http://192.168.1.1/phishing',  # IP-based
        'http://amazon-verify.xyz/account',  # Fake domain
        'http://microsoft-update.ml/download',  # Malicious TLD
        'http://facebook-security.cf/verify',  # Suspicious
        'http://secure-paypaI.com/login',  # Homoglyph (I vs l)
        'http://apple.com-verify.tk/account',  # Combosquatting
        'http://account-netflix.com/billing',  # Fake pattern
        'http://login.chase.phishing.com/secure',  # Subdomain attack
        'http://wellsfargo.com.scamsite.ru/login',  # Combosquatting
        'http://bankofamerica-alert.xyz/verify',  # Phishing
        'http://amazon.com.verify.tk/account',  # Chain attack
        'http://support-microsoft.ml/ticket',  # Fake support
        'http://instagram.secure.cf/login',  # Subdomain phishing
    ]


def test_raw_model(model_path: str, preprocessor_path: str, test_urls: list):
    """Test raw model without enhancements"""
    print("Loading raw model...")
    model = load_model(model_path, compile=False)
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    class_names = ['benign', 'defacement', 'malware', 'phishing']
    results = []
    
    for url in test_urls:
        url_seq = preprocessor.encode_urls([url])
        domain_seq = preprocessor.encode_domains([url])
        probs = model.predict([url_seq, domain_seq], verbose=0)[0]
        
        prediction = class_names[np.argmax(probs)]
        confidence = float(np.max(probs))
        
        results.append({
            'url': url,
            'prediction': prediction,
            'confidence': confidence,
            'benign_prob': float(probs[0]),
            'phishing_prob': float(probs[3]),
        })
    
    return results


def analyze_results(raw_results: list, enhanced_results: list, ground_truth: str):
    """
    Analyze and compare raw vs enhanced predictions
    
    Args:
        raw_results: Raw model predictions
        enhanced_results: Enhanced predictions
        ground_truth: Expected label ('benign' or 'malicious')
    """
    n_samples = len(raw_results)
    
    # Count false positives (legitimate URLs flagged as malicious)
    if ground_truth == 'benign':
        raw_fp = sum(1 for r in raw_results if r['prediction'] != 'benign')
        enhanced_fp = sum(1 for r in enhanced_results if r['adjusted_prediction'] != 'benign')
        
        raw_fp_rate = raw_fp / n_samples * 100
        enhanced_fp_rate = enhanced_fp / n_samples * 100
        
        print(f"\n{'='*80}")
        print(f"FALSE POSITIVE ANALYSIS (Legitimate Brands)")
        print(f"{'='*80}")
        print(f"Total legitimate URLs tested: {n_samples}")
        print(f"\nRaw Model:")
        print(f"  False Positives: {raw_fp}/{n_samples}")
        print(f"  False Positive Rate: {raw_fp_rate:.1f}%")
        print(f"\nEnhanced Model:")
        print(f"  False Positives: {enhanced_fp}/{n_samples}")
        print(f"  False Positive Rate: {enhanced_fp_rate:.1f}%")
        print(f"\nImprovement:")
        print(f"  FP Reduction: {raw_fp - enhanced_fp} fewer false positives")
        print(f"  Relative Reduction: {(1 - enhanced_fp_rate/raw_fp_rate)*100:.1f}%")
        
        # Show corrected predictions
        print(f"\n{'='*80}")
        print(f"CORRECTED PREDICTIONS (Raw → Enhanced)")
        print(f"{'='*80}")
        corrections = 0
        for raw, enh in zip(raw_results, enhanced_results):
            if raw['prediction'] != 'benign' and enh['adjusted_prediction'] == 'benign':
                corrections += 1
                domain = enh['url'].split('/')[2]
                print(f"{corrections:2d}. {domain:30s} | Raw: {raw['prediction']:12s} ({raw['confidence']:.1%}) → Enhanced: benign")
        
        print(f"\nTotal corrections: {corrections}")
        
    # Count false negatives (malicious URLs flagged as benign)
    elif ground_truth == 'malicious':
        raw_tp = sum(1 for r in raw_results if r['prediction'] != 'benign')
        enhanced_tp = sum(1 for r in enhanced_results if r['adjusted_prediction'] != 'benign')
        
        raw_detection = raw_tp / n_samples * 100
        enhanced_detection = enhanced_tp / n_samples * 100
        
        print(f"\n{'='*80}")
        print(f"MALICIOUS DETECTION ANALYSIS")
        print(f"{'='*80}")
        print(f"Total malicious URLs tested: {n_samples}")
        print(f"\nRaw Model:")
        print(f"  Detected: {raw_tp}/{n_samples}")
        print(f"  Detection Rate: {raw_detection:.1f}%")
        print(f"\nEnhanced Model:")
        print(f"  Detected: {enhanced_tp}/{n_samples}")
        print(f"  Detection Rate: {enhanced_detection:.1f}%")
        print(f"\nImpact:")
        if enhanced_detection >= raw_detection:
            print(f"  ✓ No degradation in detection rate")
        else:
            print(f"  ⚠ Detection rate decreased by {raw_detection - enhanced_detection:.1f}%")


def main():
    """Run comprehensive comparison test"""
    
    # Configuration
    MODEL_PATH = 'models/url_detector_improved.h5'
    PREPROCESSOR_PATH = 'models/preprocessor_improved.pkl'
    
    print("="*80)
    print("ENHANCED INFERENCE SYSTEM - COMPREHENSIVE TEST")
    print("="*80)
    
    # Load test data
    brand_urls = load_brand_domains()
    malicious_urls = load_malicious_urls()
    
    print(f"\nTest Set:")
    print(f"  Legitimate brands: {len(brand_urls)}")
    print(f"  Malicious URLs: {len(malicious_urls)}")
    
    # Test 1: Brand domains (FP reduction)
    print(f"\n{'='*80}")
    print("TEST 1: LEGITIMATE BRAND DOMAINS")
    print(f"{'='*80}")
    
    print("\n[1/2] Testing raw model...")
    raw_brand_results = test_raw_model(MODEL_PATH, PREPROCESSOR_PATH, brand_urls)
    
    print("[2/2] Testing enhanced model...")
    enhanced_predictor = EnhancedPredictor(MODEL_PATH, PREPROCESSOR_PATH)
    enhanced_brand_results = enhanced_predictor.batch_predict(brand_urls, return_metadata=True)
    
    analyze_results(raw_brand_results, enhanced_brand_results, 'benign')
    
    # Test 2: Malicious URLs (detection rate)
    print(f"\n{'='*80}")
    print("TEST 2: MALICIOUS URLS")
    print(f"{'='*80}")
    
    print("\n[1/2] Testing raw model...")
    raw_malicious_results = test_raw_model(MODEL_PATH, PREPROCESSOR_PATH, malicious_urls)
    
    print("[2/2] Testing enhanced model...")
    enhanced_malicious_results = enhanced_predictor.batch_predict(malicious_urls, return_metadata=True)
    
    analyze_results(raw_malicious_results, enhanced_malicious_results, 'malicious')
    
    # Inference time analysis
    print(f"\n{'='*80}")
    print("INFERENCE TIME ANALYSIS")
    print(f"{'='*80}")
    
    inference_times = [r['inference_time_ms'] for r in enhanced_brand_results]
    avg_time = np.mean(inference_times)
    max_time = np.max(inference_times)
    
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Maximum inference time: {max_time:.2f} ms")
    print(f"Target: <20ms | Status: {'✓ PASS' if avg_time < 20 else '✗ FAIL'}")
    
    # Sample detailed predictions
    print(f"\n{'='*80}")
    print("SAMPLE DETAILED PREDICTIONS")
    print(f"{'='*80}")
    
    samples = [
        ('https://google.com/search', 'High-reputation domain'),
        ('https://capitalone.com/login', 'Financial institution'),
        ('http://paypal-secure.tk/login', 'Phishing URL'),
    ]
    
    for url, description in samples:
        result = enhanced_predictor.enhanced_predict(url, return_metadata=True)
        print(f"\n{description}:")
        print(f"  URL: {url}")
        print(f"  Raw prediction: {result['raw_prediction']} ({result['raw_confidence']:.1%})")
        print(f"  Enhanced prediction: {result['adjusted_prediction']} ({result['adjusted_confidence']:.1%})")
        print(f"  Entropy: {result['entropy']:.3f}")
        if 'metadata' in result:
            print(f"  Reputation: {result['metadata']['reputation']:.2f}")
            print(f"  Decision threshold: {result['metadata']['decision_threshold']:.2f}")
            print(f"  Max malicious prob: {result['metadata']['max_malicious_prob']:.2f}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
