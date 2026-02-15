# Model Improvements - Quick Start Guide

## 🎯 What Changed?

Your URL detection model has been upgraded with a **multi-input architecture** to fix false positives on legitimate domains like GitHub, Amazon, and Python.org.

### Original Model Issue
```
github.com → 95% PHISHING ❌
amazon.com → 52% PHISHING ❌  
python.org → 98% PHISHING ❌
```

### Improved Model (Expected)
```
github.com → 85% BENIGN ✓
amazon.com → 90% BENIGN ✓
python.org → 88% BENIGN ✓
```

## 🚀 Quick Start

### 1. Train the Improved Model

```bash
python train_improved_model.py
```

This will:
- Load the same dataset (malicious_phish.csv)
- Train the multi-input architecture with focal loss
- Save model to `models/url_detector_improved.h5`
- Generate training visualizations

**Training time:** ~2-3 hours on CPU (similar to original)

### 2. Test the Improved Model

```bash
# Run full test suite
python test_improved_model.py

# Test a single URL
python test_improved_model.py --url "https://github.com/user/repo"
```

### 3. Compare Performance

```bash
# Test with original model
python main.py --predict "https://github.com"

# Test with improved model  
python test_improved_model.py --url "https://github.com"
```

## 📊 What's New?

### Architecture Changes

| Feature | Original | Improved |
|---------|----------|----------|
| **Inputs** | 1 (full URL) | 2 (URL + Domain) |
| **Branches** | Single CNN | Dual CNN (URL + Domain) |
| **Loss** | Cross-Entropy | Focal Loss |
| **Parameters** | 345K | 355K (+2.9%) |
| **Domain Extraction** | No | Yes |
| **False Positive Rate** | ~40% on brands | ~10% (expected) |

### Why It Works Better

1. **Domain Branch**: Learns "github.com is usually benign" from training data
2. **URL Branch**: Still catches malicious patterns in paths/parameters
3. **Fusion**: Combines both signals - weighs domain trust vs URL-level threats
4. **Focal Loss**: Reduces overconfident predictions on ambiguous cases

## 📁 New Files

### Core Implementation
- `src/model.py` - Added `build_multi_input_cnn_model()` and `focal_loss()`
- `src/preprocess.py` - Added `extract_domain()` and `encode_domains()`
- `src/train.py` - Updated to support multi-input training

### Training & Testing
- `train_improved_model.py` - Train improved model
- `test_improved_model.py` - Test improved model with challenge URLs
- `IMPROVED_MODEL_DOCUMENTATION.md` - Comprehensive technical documentation

### Model Artifacts (after training)
- `models/url_detector_improved.h5` - Trained improved model
- `models/preprocessor_improved.pkl` - Preprocessor with domain extraction
- `models/training_history_improved.png` - Training curves
- `models/training_metadata_improved.json` - Training statistics

## 🔍 Technical Deep Dive

### Multi-Input Architecture

```
Input 1: Full URL           Input 2: Domain Only
    |                              |
Embedding(128D)             Embedding(64D)
    |                              |
3× Conv1D Blocks            2× Conv1D Blocks
    |                              |
64 features                 64 features
    |                              |
    +---------- Concat ------------+
                   |
           Dense(128) → Dense(64)
                   |
              Output(4 classes)
```

### Domain Extraction Examples

```python
extract_domain("https://github.com/user/repo")           → "github.com"
extract_domain("http://bankofamerica-verify.tk/login")   → "bankofamerica-verify.tk"
extract_domain("192.168.1.1/admin")                      → "192.168.1.1"
```

### Focal Loss Formula

```
FL(p_t) = -α(1 - p_t)^γ log(p_t)

where:
  p_t = predicted probability for true class
  γ = 2.0 (focusing parameter)
  α = 0.25 (balancing factor)
  
Effect: Down-weights easy examples, focuses on hard cases
```

## 📈 Expected Performance

| Metric | Original | Improved (Target) |
|--------|----------|-------------------|
| Overall Accuracy | 97.78% | ≥97.5% |
| Legitimate Site FP | ~40% | <10% |
| Malware Detection | 96.36% | ≥96% |
| Phishing Detection | 96.20% | ≥96% |
| Parameters | 345K | 355K |

## 🛠️ Usage in Production

### Load and Use Improved Model

```python
import tensorflow as tf
from src.preprocess import URLPreprocessor

# Load model and preprocessor
model = tf.keras.models.load_model('models/url_detector_improved.h5')
preprocessor = URLPreprocessor.load('models/preprocessor_improved.pkl')

# Predict URL
url = "https://example.com/suspicious/path"
url_seq = preprocessor.encode_urls([url])
domain_seq = preprocessor.encode_domains([url])

probs = model.predict([url_seq, domain_seq])[0]
pred_class_idx = probs.argmax()
pred_class = preprocessor.decode_label(pred_class_idx)

print(f"Predicted: {pred_class} ({probs.max():.2%} confidence)")
```

### Integration Tips

1. **Confidence Thresholds:**
   - High risk: >90% malicious confidence
   - Medium risk: 70-90% (flag for review)
   - Low risk: <70% (allow with logging)

2. **Domain Whitelist (Optional):**
   ```python
   TRUSTED_DOMAINS = {'github.com', 'google.com', 'microsoft.com', ...}
   
   if domain in TRUSTED_DOMAINS and confidence < 0.95:
       # Reduce malicious probability for known-good domains
       # Only flag if extremely high confidence
   ```

3. **Fallback Strategy:**
   ```python
   # Use improved model as primary
   result_improved = improved_model.predict([url_seq, domain_seq])
   
   # If uncertain, consult original model
   if result_improved.max() < 0.8:
       result_original = original_model.predict(url_seq)
       # Combine predictions or flag for manual review
   ```

## 🔬 Validation Checklist

After training, verify improvements:

- [ ] Train improved model successfully
- [ ] Test on GitHub URLs → should predict BENIGN
- [ ] Test on Amazon URLs → should predict BENIGN
- [ ] Test on Python.org → should predict BENIGN
- [ ] Test on obvious phishing → should still catch it
- [ ] Test on malware downloads → should still catch it
- [ ] Overall accuracy ≥97%
- [ ] False positive rate on brands <15%

## 📚 Additional Resources

- **Full Technical Documentation:** [IMPROVED_MODEL_DOCUMENTATION.md](IMPROVED_MODEL_DOCUMENTATION.md)
- **Original Project README:** [README.md](README.md)
- **Architecture Details:** See `src/model.py` docstrings
- **Preprocessing Logic:** See `src/preprocess.py` docstrings

## ❓ FAQ

**Q: Do I need to retrain from scratch?**  
A: Yes, the architecture changed (single-input → multi-input). Existing weights aren't compatible.

**Q: Will this take longer to train?**  
A: Only slightly (~10% more due to domain branch). Still ~2-3 hours on CPU.

**Q: Can I use the original model too?**  
A: Yes! Both models can coexist. Original at `url_detector.h5`, improved at `url_detector_improved.h5`.

**Q: What if improved model performs worse?**  
A: Keep the original! The improved model targets specific false positive issues. If your use case doesn't have this problem, original is fine.

**Q: How do I know which model to use?**  
A: If you see false positives on legitimate brands → use improved. If not → original is simpler and proven.

## 🤝 Support

Issues or questions about the improvements?
1. Check [IMPROVED_MODEL_DOCUMENTATION.md](IMPROVED_MODEL_DOCUMENTATION.md) for technical details
2. Review training logs in `models/training_metadata_improved.json`
3. Test specific URLs with `test_improved_model.py --url <url>`

---

**Version:** 1.0  
**Date:** February 11, 2026  
**Compatibility:** TensorFlow 2.15+, Python 3.12+
