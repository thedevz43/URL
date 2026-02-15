"""
Run Comprehensive Stress Testing + Brand Bias Testing

Executes:
1. Full stress testing suite (6 categories)
2. Detailed brand domain testing (50 brands)
3. Generates combined report
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow import keras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model import focal_loss
from src.adversarial_generators import BrandDomainGenerator
from test_suite import ComprehensiveStressTest

print("="*80)
print("COMPREHENSIVE STRESS TESTING + BRAND BIAS EVALUATION")
print("="*80)
print()

# Configuration
MODEL_PATH = 'models/url_detector_augmented.h5'
PREPROCESSOR_PATH = 'models/preprocessor_augmented.pkl'
DATA_PATH = 'data/malicious_phish_augmented.csv'

print("Using augmented model:")
print(f"  Model: {MODEL_PATH}")
print(f"  Preprocessor: {PREPROCESSOR_PATH}")
print(f"  Dataset: {DATA_PATH}")
print()

# ============================================================================
# PART 1: COMPREHENSIVE STRESS TESTING
# ============================================================================

print("="*80)
print("PART 1: COMPREHENSIVE STRESS TESTING (6 SUITES)")
print("="*80)
print()

stress_test = ComprehensiveStressTest(
    model_path=MODEL_PATH,
    preprocessor_path=PREPROCESSOR_PATH,
    data_path=DATA_PATH
)

print("\n[1/6] Running Temporal Testing...")
temporal_results = stress_test.run_temporal_test()
print(f"✓ Temporal test complete")

print("\n[2/6] Running Adversarial Testing (200+ attacks)...")
adversarial_results = stress_test.run_adversarial_test(n_samples=200)
print(f"✓ Adversarial test complete")

print("\n[3/6] Running Brand Bias Testing (50 brands)...")
brand_results = stress_test.run_brand_bias_test()
brand_fp_rate = brand_results.get('false_positive_rate', 0)
print(f"✓ Brand bias test complete: {brand_fp_rate*100:.1f}% FP rate")

print("\n[4/6] Running Robustness Testing (86 edge cases)...")
robustness_results = stress_test.run_robustness_test()
print(f"✓ Robustness test complete")

print("\n[5/6] Running Calibration Analysis...")
calibration_results = stress_test.run_calibration_analysis()
ece = calibration_results.get('ece', 0)
print(f"✓ Calibration complete: ECE = {ece:.4f}")

print("\n[6/6] Running OOD Detection...")
ood_results = stress_test.run_ood_detection()
print(f"✓ OOD detection complete")

# Generate full report
print("\n" + "="*80)
print("Generating comprehensive report...")
report = stress_test.generate_report(
    save_path='models/comprehensive_test_results.json'
)
print(f"✓ Report saved to models/comprehensive_test_results.json")

# ============================================================================
# PART 2: DETAILED BRAND DOMAIN TESTING
# ============================================================================

print("\n" + "="*80)
print("PART 2: DETAILED BRAND DOMAIN TESTING")
print("="*80)
print()

# Load model and preprocessor
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={'focal_loss_fixed': focal_loss()}
)

with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)

# Generate brand test URLs
brand_gen = BrandDomainGenerator()
brand_urls = []
brand_patterns = [
    lambda d: f"https://{d}",
    lambda d: f"https://www.{d}",
    lambda d: f"https://{d}/login",
    lambda d: f"https://www.{d}/account",
    lambda d: f"https://{d}/docs",
    lambda d: f"https://api.{d}/v1",
    lambda d: f"https://{d}/products",
    lambda d: f"https://support.{d}",
]

print("Generating brand test URLs...")
for domain in brand_gen.TOP_DOMAINS[:50]:  # Top 50 brands
    for pattern in brand_patterns:
        brand_urls.append({
            'url': pattern(domain),
            'domain': domain,
            'true_label': 'benign'
        })

print(f"Generated {len(brand_urls)} brand test URLs from 50 domains")
print()

# Test brand URLs
print("Testing brand URLs...")
results = []

for item in brand_urls:
    url = item['url']
    
    # Encode
    url_seq = preprocessor.encode_urls([url])
    domain_seq = preprocessor.encode_domains([url])
    
    # Predict
    pred_probs = model.predict([url_seq, domain_seq], verbose=0)
    pred_class = np.argmax(pred_probs[0])
    confidence = np.max(pred_probs[0])
    pred_label = preprocessor.label_encoder.classes_[pred_class]
    
    is_fp = (pred_label != 'benign')
    
    results.append({
        'url': url,
        'domain': item['domain'],
        'true_label': 'benign',
        'predicted_label': pred_label,
        'confidence': float(confidence),
        'is_false_positive': is_fp,
        'probabilities': {
            preprocessor.label_encoder.classes_[i]: float(pred_probs[0][i])
            for i in range(len(pred_probs[0]))
        }
    })

# Calculate metrics
total = len(results)
false_positives = sum(1 for r in results if r['is_false_positive'])
fp_rate = false_positives / total

# Per-domain analysis
domain_stats = {}
for result in results:
    domain = result['domain']
    if domain not in domain_stats:
        domain_stats[domain] = {
            'total': 0,
            'false_positives': 0,
            'predictions': []
        }
    
    domain_stats[domain]['total'] += 1
    if result['is_false_positive']:
        domain_stats[domain]['false_positives'] += 1
    domain_stats[domain]['predictions'].append(result['predicted_label'])

# Calculate per-domain FP rates
for domain in domain_stats:
    stats = domain_stats[domain]
    stats['fp_rate'] = stats['false_positives'] / stats['total']

# Sort domains by FP rate
problematic_domains = sorted(
    domain_stats.items(),
    key=lambda x: x[1]['fp_rate'],
    reverse=True
)

print("\n" + "="*80)
print("DETAILED BRAND TESTING RESULTS")
print("="*80)
print()
print(f"Total URLs tested: {total}")
print(f"False Positives: {false_positives}")
print(f"False Positive Rate: {fp_rate*100:.2f}%")
print()

print("Top 10 Most Problematic Domains:")
print("-" * 80)
for i, (domain, stats) in enumerate(problematic_domains[:10], 1):
    if stats['fp_rate'] > 0:
        print(f"{i:2}. {domain:30} : {stats['fp_rate']*100:5.1f}% FP ({stats['false_positives']}/{stats['total']})")

print()
print("Domains with 0% False Positives:")
print("-" * 80)
perfect_domains = [d for d, s in problematic_domains if s['fp_rate'] == 0]
print(f"Count: {len(perfect_domains)}/{len(domain_stats)}")
if perfect_domains:
    for domain in perfect_domains[:20]:
        print(f"  ✓ {domain}")

# Save detailed results
brand_report = {
    'timestamp': datetime.now().isoformat(),
    'model': MODEL_PATH,
    'summary': {
        'total_urls': total,
        'total_domains': len(domain_stats),
        'false_positives': false_positives,
        'false_positive_rate': float(fp_rate),
        'perfect_domains': len(perfect_domains)
    },
    'per_domain_stats': {
        domain: {
            'total': stats['total'],
            'false_positives': stats['false_positives'],
            'fp_rate': float(stats['fp_rate']),
            'predictions': stats['predictions']
        }
        for domain, stats in domain_stats.items()
    },
    'detailed_results': results[:100]  # First 100 for size
}

with open('models/detailed_brand_test_results.json', 'w') as f:
    json.dump(brand_report, f, indent=2)

print()
print("✓ Detailed brand results saved to models/detailed_brand_test_results.json")

# ============================================================================
# COMBINED SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMBINED SUMMARY")
print("="*80)
print()

summary = {
    'stress_testing': {
        'temporal_stability': temporal_results.get('performance_stability', 'N/A'),
        'adversarial_robustness': f"{adversarial_results.get('detection_rate', 0)*100:.1f}%",
        'brand_fp_rate': f"{brand_fp_rate*100:.1f}%",
        'robustness_score': f"{robustness_results.get('safe_percent', 0)*100:.1f}%",
        'calibration_ece': f"{ece:.4f}",
        'ood_detection': f"{ood_results.get('avg_confidence_decrease', 0)*100:.1f}%"
    },
    'detailed_brand_testing': {
        'total_urls': total,
        'total_domains': len(domain_stats),
        'overall_fp_rate': f"{fp_rate*100:.2f}%",
        'perfect_domains': len(perfect_domains),
        'problematic_domains': len([d for d in domain_stats.values() if d['fp_rate'] > 0.2])
    }
}

print("STRESS TESTING RESULTS:")
print("-" * 80)
for key, value in summary['stress_testing'].items():
    print(f"  {key:25} : {value}")

print()
print("BRAND TESTING RESULTS:")
print("-" * 80)
for key, value in summary['detailed_brand_testing'].items():
    print(f"  {key:25} : {value}")

print()
print("="*80)
print("ALL TESTS COMPLETE!")
print("="*80)
print()
print("Reports generated:")
print("  - models/comprehensive_test_results.json")
print("  - COMPREHENSIVE_TEST_RESULTS.md")
print("  - models/detailed_brand_test_results.json")
print()
