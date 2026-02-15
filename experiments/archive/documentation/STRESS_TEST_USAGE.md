# Stress Testing Framework - Usage Guide

## Overview

This comprehensive stress-testing framework evaluates model robustness across 6 critical dimensions:

1. **Temporal Degradation** - Performance on future data
2. **Adversarial URLs** - Resistance to attack patterns
3. **Brand Bias** - False positives on legitimate sites
4. **Robustness** - Edge case handling
5. **Confidence Calibration** - Reliability of predictions
6. **OOD Detection** - Identifying unusual inputs

## Quick Start

### Run Complete Test Suite

```bash
python test_suite.py
```

This runs all 6 tests and generates:
- `models/stress_test_report.json` - Detailed results
- `models/stress_test_calibration.png` - Calibration plots
- Console output with executive summary

**Expected runtime:** 10-15 minutes

---

## Individual Test Modules

### 1. Generate Adversarial URLs

```python
from src.adversarial_generators import AdversarialURLGenerator

gen = AdversarialURLGenerator(seed=42)

# Generate specific attack type
homoglyph_attack = gen.generate_homoglyph_attack('paypal')
print(homoglyph_attack)
# Output: {
#   'url': 'https://pаypal.com/login',  # Cyrillic 'а'
#   'type': 'homoglyph',
#   'target_brand': 'paypal',
#   'substitutions': 1
# }

# Generate batch of 100 adversarial URLs
batch = gen.generate_adversarial_batch(100)

# Attack types included:
# - Homoglyph attacks (visual character substitution)
# - Subdomain chains (hide malicious domain)
# - Typosquatting (common typos)
# - Combosquatting (brand + keyword)
# - IDN homograph (mixed scripts)
# - IP obfuscation(decimal, octal, hex)
# - Fake subdomains
# - Fake URL shorteners
```

### 2. Generate Robustness Test Cases

```python
from src.robustness_tests import RobustnessTestGenerator

gen = RobustnessTestGenerator(seed=42)

# Generate specific edge case types
empty_strings = gen.generate_empty_strings()
long_urls = gen.generate_extremely_long_urls()
unicode_urls = gen.generate_unicode_urls()
malformed = gen.generate_malformed_urls()

# Or generate complete suite (86 test cases)
all_tests = gen.generate_complete_robustness_suite()

print(f"Total edge cases: {len(all_tests)}")
for test in all_tests[:5]:
    print(f"  {test['type']:20} - {test['description']}")
```

### 3. Test Model Robustness

```python
from src.robustness_tests import test_model_robustness

# Test model on edge cases
results = test_model_robustness(model, preprocessor, all_tests)

print(f"Success rate: {results['success_rate']:.2%}")
print(f"Errors: {results['errors']}")

# Check for crashes
if results['crashes']:
    print("\nCrashed on:")
    for crash in results['crashes']:
        print(f"  {crash['type']}: {crash['error']}")
```

### 4. Confidence Calibration

```python
from src.calibration import ConfidenceCalibrator

calibrator = ConfidenceCalibrator()

# Compute Expected Calibration Error
ece_results = calibrator.compute_ece(y_true, y_pred_probs, n_bins=15)
print(f"ECE: {ece_results['ECE']:.4f}")
print(f"MCE: {ece_results['MCE']:.4f}")

# Plot reliability diagram
calibrator.plot_reliability_diagram(
    y_true, y_pred_probs,
    save_path='calibration_plot.png'
)

# Analyze confidence distribution
conf_dist = calibrator.analyze_confidence_distribution(y_pred_probs)
print(f"High confidence (>90%): {conf_dist['high_confidence_rate']:.2%}")
print(f"Low confidence (<50%): {conf_dist['low_confidence_rate']:.2%}")

# Temperature scaling (if calibration is poor)
optimal_temp = calibrator.fit_temperature(logits, y_true)
print(f"Optimal temperature: {optimal_temp:.2f}")

calibrated_probs = calibrator.apply_temperature(logits)
```

### 5. Out-of-Distribution Detection

```python
from src.calibration import OutOfDistributionDetector

ood_detector = OutOfDistributionDetector(
    entropy_threshold=1.0,
    confidence_threshold=0.5
)

# Detect OOD samples
ood_results = ood_detector.detect_ood(y_pred_probs)

print(f"OOD rate: {ood_results['ood_rate']:.2%}")
print(f"High entropy: {ood_results['high_entropy_rate']:.2%}")
print(f"Low confidence: {ood_results['low_confidence_rate']:.2%}")

# Get indices of OOD samples
ood_indices = ood_results['ood_indices']
print(f"OOD samples at indices: {ood_indices[:10]}")

# Compute entropy and uncertainty
entropy = ood_detector.compute_entropy(y_pred_probs)
uncertainty_analysis = ood_detector.analyze_uncertainty(y_pred_probs)
```

---

## Custom Testing Scenarios

### Test Specific Brand Domains

```python
from src.adversarial_generators import BrandDomainGenerator

brand_gen = BrandDomainGenerator()

# Generate legitimate URLs from top domains
legit_urls = brand_gen.generate_legitimate_urls(n_samples=100)

# Test for false positives
fp_count = 0
for url_data in legit_urls:
    url = url_data['url']
    
    # Your model prediction code
    prediction = model.predict(url)
    
    if prediction != 'benign':
        fp_count += 1
        print(f"FALSE POSITIVE: {url_data['domain']} → {prediction}")

fp_rate = fp_count / len(legit_urls)
print(f"\nFalse Positive Rate: {fp_rate:.2%}")
```

### Test Custom URL List

```python
# Test custom list of URLs
custom_urls = [
    'https://github.com/user/repo',
    'https://amazon.com/product/B08N5WRWNW',
    'http://bit.ly/suspicious',
]

for url in custom_urls:
    url_seq = preprocessor.encode_urls([url])
    domain_seq = preprocessor.encode_domains([url])
    pred_probs = model.predict([url_seq, domain_seq], verbose=0)[0]
    pred_class = class_names[np.argmax(pred_probs)]
    confidence = pred_probs.max()
    
    print(f"{url:50} → {pred_class:12} ({confidence:.2%})")
```

### Temporal Testing on Custom Data

```python
# Load your dataset with timestamps
df = pd.read_csv('your_data.csv')

# Split by date (assuming 'timestamp' column)
cutoff_date = '2025-01-01'
old_data = df[df['timestamp'] < cutoff_date]
new_data = df[df['timestamp'] >= cutoff_date]

# Evaluate on both
old_accuracy = evaluate(model, old_data)
new_accuracy = evaluate(model, new_data)

degradation = old_accuracy - new_accuracy
print(f"Temporal degradation: {degradation:.2%}")
```

---

## Interpreting Results

### Expected Calibration Error (ECE)

| ECE Range | Interpretation |
|-----------|----------------|
| 0.00 - 0.05 | Excellent calibration |
| 0.05 - 0.10 | Good calibration |
| 0.10 - 0.20 | Poor calibration (temperature scaling recommended) |
| 0.20+ | Severe miscalibration (model confidence unreliable) |

### False Positive Rate Thresholds

| FP Rate | Interpretation | Action |
|---------|----------------|--------|
| 0-5% | Acceptable for production | Deploy |
| 5-10% | Elevated but manageable | Monitor closely |
| 10-20% | High - review needed | Investigate root causes |
| 20%+ | Unacceptable | Do not deploy; retrain |

### Temporal Degradation

| Degradation | Interpretation | Action |
|-------------|----------------|--------|
| 0-2% | Excellent stability | Monitor quarterly |
| 2-5% | Good stability | Monitor monthly |
| 5-10% | Moderate drift | Consider retraining |
| 10%+ | High drift | Retrain immediately |

### Adversarial Detection

| Detection Rate | Interpretation | Security Risk |
|----------------|----------------|---------------|
| 95-100% | Excellent | Low |
| 90-95% | Good | Medium |
| 80-90% | Moderate | High |
| <80% | Poor | Critical |

---

## Production Deployment Checklist

### Pre-Deployment Tests

- [ ] Run complete stress test suite
- [ ] ECE < 0.10 (good calibration)
- [ ] False positive rate < 10% on brand domains
- [ ] Adversarial detection > 95%
- [ ] Temporal degradation < 5%
- [ ] Robustness success rate > 95%
- [ ] Input validation implemented
- [ ] Confidence thresholds configured

### Deployment Safeguards

```python
def production_predict(url):
    """
    Production-ready prediction with safeguards
    """
    # 1. Input validation
    if not isinstance(url, str):
        raise ValueError("URL must be string")
    
    if len(url) == 0:
        raise ValueError("URL cannot be empty")
    
    # 2. Domain whitelist check
    domain = extract_domain(url)
    if domain in TRUSTED_DOMAINS:
        return {
            'class': 'benign',
            'confidence': 1.0,
            'source': 'whitelist'
        }
    
    # 3. Model prediction
    url_seq = preprocessor.encode_urls([url])
    domain_seq = preprocessor.encode_domains([url])
    pred_probs = model.predict([url_seq, domain_seq], verbose=0)[0]
    
    # 4. OOD detection
    entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-12))
    confidence = pred_probs.max()
    
    if entropy > 1.0 or confidence < 0.5:
        return {
            'class': 'uncertain',
            'confidence': confidence,
            'entropy': entropy,
            'source': 'ood_flagged'
        }
    
    # 5. Confidence threshold
    pred_class = class_names[np.argmax(pred_probs)]
    
    if pred_class != 'benign' and confidence < 0.70:
        return {
            'class': 'uncertain',
            'confidence': confidence,
            'source': 'low_confidence'
        }
    
    return {
        'class': pred_class,
        'confidence': float(confidence),
        'all_probs': pred_probs.tolist(),
        'source': 'model'
    }
```

### Monitoring in Production

```python
# Log predictions for monitoring
def log_prediction(url, prediction, user_feedback=None):
    """
    Log all predictions for drift detection
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'url': url[:100],  # Truncate for storage
        'domain': extract_domain(url),
        'prediction': prediction['class'],
        'confidence': prediction['confidence'],
        'source': prediction['source'],
        'user_feedback': user_feedback  # If user reports false positive
    }
    
    # Send to logging system (e.g., CloudWatch, Datadog)
    logger.info(json.dumps(log_entry))
    
    # Check for anomalies
    if prediction['confidence'] < 0.5:
        alert_low_confidence(log_entry)
    
    if user_feedback == 'false_positive':
        alert_false_positive(log_entry)

# weekly monitoring metrics
def compute_weekly_metrics():
    logs = load_logs_from_past_week()
    
    metrics = {
        'total_predictions': len(logs),
        'avg_confidence': np.mean([l['confidence'] for l in logs]),
        'low_confidence_rate': np.mean([l['confidence'] < 0.5 for l in logs]),
        'class_distribution': Counter([l['prediction'] for l in logs]),
        'false_positive_rate': compute_fp_rate(logs),
        'false_negative_rate': compute_fn_rate(logs)
    }
    
    # Alert if metrics degrade
    if metrics['false_positive_rate'] > 0.10:
        send_alert("High false positive rate detected")
    
    if metrics['avg_confidence'] < 0.70:
        send_alert("Low average confidence - possible drift")
    
    return metrics
```

---

## Troubleshooting

### Issue: High False Positive Rate

**Symptom:** Legitimate URLs classified as malicious

**Diagnosis:**
```python
# Test on brand domains
brand_gen = BrandDomainGenerator()
legit_urls = brand_gen.generate_legitimate_urls(50)

# Analyze false positives
for url_data in legit_urls:
    prediction = predict(url_data['url'])
    if prediction != 'benign':
        print(f"FP: {url_data['domain']} → {prediction}")
```

**Solutions:**
1. Implement domain whitelist
2. Collect more legitimate brand URLs for retraining
3. Increase confidence threshold to 0.70 or 0.80

### Issue: Model Crashes on Edge Cases

**Symptom:** TypeError or ValueError on certain inputs

**Diagnosis:**
```python
from src.robustness_tests import RobustnessTestGenerator, test_model_robustness

gen = RobustnessTestGenerator()
tests = gen.generate_complete_robustness_suite()
results = test_model_robustness(model, preprocessor, tests)

# Check crashes
for crash in results['crashes']:
    print(f"{crash['type']}: {crash['error']}")
```

**Solutions:**
1. Add input validation (check type, length)
2. Handle empty strings explicitly
3. Catch exceptions in preprocessing pipeline

### Issue: Poor Calibration (High ECE)

**Symptom:** Confidence doesn't match accuracy

**Diagnosis:**
```python
from src.calibration import ConfidenceCalibrator

calibrator = ConfidenceCalibrator()
ece_results = calibrator.compute_ece(y_true, y_pred_probs)
print(f"ECE: {ece_results['ECE']:.4f}")

# Plot reliability diagram
calibrator.plot_reliability_diagram(y_true, y_pred_probs)
```

**Solutions:**
1. Apply temperature scaling:
   ```python
   optimal_T = calibrator.fit_temperature(logits, y_true)
   calibrated_probs = calibrator.apply_temperature(logits)
   ```
2. Retrain with focal loss or label smoothing
3. Use Platt scaling for extreme miscalibration

---

## Advanced Usage

### Custom Attack Patterns

```python
# Define your own adversarial patterns
custom_patterns = [
    {
        'url': 'https://gооgle.com/signin',  # Cyrillic о
        'expected': 'phishing',
        'description': 'Homoglyph attack on Google'
    },
    {
        'url': 'https://paypal-security-verify.tk/account',
        'expected': 'phishing',
        'description': 'Combosquatting with suspicious TLD'
    }
]

# Test detection
for pattern in custom_patterns:
    prediction = model.predict(pattern['url'])
    status = "✓" if prediction == pattern['expected'] else "✗"
    print(f"{status} {pattern['description']}")
```

### Batch Testing

```python
# Test large batch efficiently
urls = load_urls_from_file('test_urls.txt')

# Batch encode
url_seqs = preprocessor.encode_urls(urls)
domain_seqs = preprocessor.encode_domains(urls)

# Batch predict
pred_probs = model.predict([url_seqs, domain_seqs], batch_size=256)

# Analyze results
pred_classes = np.argmax(pred_probs, axis=1)
confidences = np.max(pred_probs, axis=1)

print(f"Average confidence: {confidences.mean():.4f}")
print(f"Low confidence (<50%): {(confidences < 0.5).mean():.2%}")
```

---

## Files Generated

### After Running test_suite.py

1. **models/stress_test_report.json**
   - Complete test results in JSON format
   - All metrics, predictions, and statistics
   - Machine-readable for further analysis

2. **models/stress_test_calibration.png**
   - Reliability diagram (confidence vs accuracy)
   - Confidence distribution histogram
   - Visual assessment of calibration quality

3. **Console Output**
   - Executive summary of all tests
   - Critical issues and warnings
   - Overall production-readiness verdict

---

## Questions?

For issues or questions about the stress testing framework:

1. Check this usage guide
2. Review `STRESS_TEST_REPORT.md` for detailed analysis
3. Examine source code in `src/adversarial_generators.py`, `src/robustness_tests.py`, `src/calibration.py`
4. Run tests with verbose output: `python test_suite.py --verbose`

---

**Framework Version:** 1.0  
**Last Updated:** February 13, 2026  
**Compatibility:** TensorFlow 2.x, Python 3.7+
