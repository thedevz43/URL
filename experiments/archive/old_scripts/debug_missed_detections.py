"""
Debug: Identify which malicious URLs are being missed and why
"""

from enhanced_inference import EnhancedPredictor

def main():
    MODEL_PATH = 'models/url_detector_improved.h5'
    PREPROCESSOR_PATH = 'models/preprocessor_improved.pkl'
    
    predictor = EnhancedPredictor(MODEL_PATH, PREPROCESSOR_PATH)
    
    malicious_urls = [
        'http://g00gle.com/signin',
        'http://paypal-secure.tk/login',
        'http://192.168.1.1/phishing',
        'http://amazon-verify.xyz/account',
        'http://microsoft-update.ml/download',
        'http://facebook-security.cf/verify',
        'http://secure-paypaI.com/login',
        'http://apple.com-verify.tk/account',
        'http://account-netflix.com/billing',
        'http://login.chase.phishing.com/secure',
        'http://wellsfargo.com.scamsite.ru/login',
        'http://bankofamerica-alert.xyz/verify',
        'http://amazon.com.verify.tk/account',
        'http://support-microsoft.ml/ticket',
        'http://instagram.secure.cf/login',
    ]
    
    print("="*80)
    print("MALICIOUS URL DETECTION ANALYSIS")
    print("="*80)
    
    missed = []
    detected = []
    
    for url in malicious_urls:
        result = predictor.enhanced_predict(url, return_metadata=True)
        
        is_detected = result['adjusted_prediction'] != 'benign'
        
        if is_detected:
            detected.append((url, result))
        else:
            missed.append((url, result))
        
        status = "✓ DETECTED" if is_detected else "✗ MISSED"
        print(f"\n{status}")
        print(f"  URL: {url}")
        print(f"  Prediction: {result['adjusted_prediction']}")
        print(f"  Max malicious prob: {result['metadata']['max_malicious_prob']:.3f}")
        print(f"  Reputation: {result['metadata']['reputation']:.2f}")
        print(f"  Threshold: {result['metadata']['decision_threshold']:.2f}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Detected: {len(detected)}/15 ({len(detected)/15*100:.1f}%)")
    print(f"Missed: {len(missed)}/15 ({len(missed)/15*100:.1f}%)")
    
    if missed:
        print(f"\n{'='*80}")
        print(f"MISSED URLS DETAILS")
        print(f"{'='*80}")
        for url, result in missed:
            print(f"\nURL: {url}")
            print(f"  Max malicious prob: {result['metadata']['max_malicious_prob']:.3f}")
            print(f"  Benign prob: {result['raw_confidences'][0]:.3f}")
            print(f"  Reputation: {result['metadata']['reputation']:.2f}")
            print(f"  Why missed: ", end="")
            
            max_mal = result['metadata']['max_malicious_prob']
            rep = result['metadata']['reputation']
            
            if max_mal < 0.50:
                print(f"Confidence too low (<50%)")
            elif max_mal < 0.75 and rep >= 0.95:
                print(f"Elite domain protection (50-75% + rep {rep:.2f})")
            elif max_mal < 0.93 and rep >= 0.95:
                print(f"Elite domain protection (75-93% + rep {rep:.2f})")
            else:
                print(f"Unknown (mal={max_mal:.2f}, rep={rep:.2f})")

if __name__ == "__main__":
    main()
