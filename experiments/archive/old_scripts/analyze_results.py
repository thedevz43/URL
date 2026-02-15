"""
Comprehensive Analysis of Stress Test Results
"""
import json

# Load report
with open('models/stress_test_report.json', 'r') as f:
    report = json.load(f)

print("="*80)
print("COMPREHENSIVE STRESS TEST RESULTS - AUGMENTED MODEL")
print("="*80)
print(f"Model: {report['model_path']}")
print(f"Timestamp: {report['timestamp']}")
print("="*80)

tests = report['tests']

# 1. TEMPORAL TEST
print("\n[1/6] TEMPORAL DEGRADATION ANALYSIS")
print("-"*80)
temporal = tests['temporal']
old_acc = temporal['old_results']['accuracy']
new_acc = temporal['new_results']['accuracy']
degradation = (old_acc - new_acc) * 100
print(f"Old Data Accuracy: {old_acc*100:.2f}%")
print(f"New Data Accuracy: {new_acc*100:.2f}%")
print(f"Performance Drop: {degradation:.2f}%")
if degradation > 2.0:
    print("  ⚠️ WARNING: Significant temporal degradation detected!")
print()

# 2. ADVERSARIAL TEST
print("[2/6] ADVERSARIAL ROBUSTNESS ANALYSIS")
print("-"*80)
adv = tests['adversarial']
total = adv['total']
detected_count = sum(v['detected_as_malicious'] for v in adv['by_type'].values())
detection_rate = detected_count / total
fnr = 1 - detection_rate

print(f"Overall Detection Rate: {detection_rate*100:.2f}%")
print(f"False Negative Rate: {fnr*100:.2f}%")
print(f"URLs Tested: {total}")
print(f"Missed Attacks: {total - detected_count}")
print("\nPerformance by Attack Type:")
for attack_type, metrics in sorted(adv['by_type'].items()):
    rate = metrics['detected_as_malicious'] / metrics['total'] * 100
    status = "❌ CRITICAL" if rate < 20 else ("⚠️ WARNING" if rate < 50 else "✅ OK")
    print(f"  {attack_type:20s}: {rate:5.1f}% [{status}]")
print()

# 3. BRAND BIAS TEST
print("[3/6] BRAND BIAS ANALYSIS")
print("-"*80)
brand = tests['brand_bias']
total = brand['total']
correct = brand['correct_benign']
fps = brand.get('false_positives', [])
fp_rate = len(fps) / total

print(f"Legitimate Brands Tested: {total}")
print(f"Correctly Classified: {correct} ({correct/total*100:.1f}%)")
print(f"False Positive Rate: {fp_rate*100:.2f}%")
print(f"\nMisclassified Brands: {len(fps)}")
if fps:
    for i, error in enumerate(fps[:10], 1):
        print(f"  {i}. {error['domain']:30s} -> {error['predicted_class']:12s} ({error['confidence']*100:.1f}%)")
print()

# 4. ROBUSTNESS TEST
print("[4/6] EDGE CASE ROBUSTNESS")
print("-"*80)
robust = tests['robustness']
total_tests = robust['total_tests']
successful = robust['successful']
errors = robust.get('errors', 0)
success_rate = successful / total_tests

print(f"Total Tests: {total_tests}")
print(f"Successful: {successful} ({success_rate*100:.2f}%)")
print(f"Crashes/Errors: {errors}")
if 'crashes' in robust and robust['crashes']:
    print("\nFailed Cases:")
    for failure in robust['crashes'][:5]:
        print(f"  - {failure['type']}: {failure['error']}")
print()

# 5. CALIBRATION ANALYSIS
print("[5/6] CALIBRATION QUALITY")
print("-"*80)
calib = tests['calibration']
print(f"Expected Calibration Error (ECE): {calib['ece']:.4f}")
print(f"Maximum Calibration Error (MCE): {calib['mce']:.4f}")
print(f"Average Confidence: {calib['avg_confidence']*100:.2f}%")
print(f"Average Accuracy: {calib['avg_accuracy']*100:.2f}%")
if calib['ece'] > 0.08:
    print("  ⚠️ WARNING: Poor calibration - model overconfident!")
elif calib['ece'] < 0.08:
    print("  ✅ Good calibration!")
print()

# 6. OOD DETECTION
print("[6/6] OUT-OF-DISTRIBUTION DETECTION")
print("-"*80)
ood = tests['ood_detection']
import numpy as np
test_entropy = np.array(ood['test_set_ood']['entropy'])
adv_entropy = np.array(ood['adversarial_ood']['entropy'])
test_ood_rate = np.mean(test_entropy > 1.0)  # Basic threshold
adv_ood_rate = np.mean(adv_entropy > 1.0)

print(f"Test Set OOD Rate: {test_ood_rate*100:.2f}%")
print(f"Adversarial OOD Rate: {adv_ood_rate*100:.2f}%")
print(f"Mean Test Entropy: {np.mean(test_entropy):.4f}")
print(f"Mean Adversarial Entropy: {np.mean(adv_entropy):.4f}")
print()

# SUMMARY OF CRITICAL ISSUES
print("="*80)
print("CRITICAL ISSUES SUMMARY")
print("="*80)

issues = []

# Temporal
old_acc = temporal['old_results']['accuracy']
new_acc = temporal['new_results']['accuracy']
degradation = (old_acc - new_acc) * 100
if degradation > 2.0:
    issues.append(f"❌ [TEMPORAL] {degradation:.1f}% degradation over time")

# Adversarial
total = adv['total']
detected_count = sum(v['detected_as_malicious'] for v in adv['by_type'].values())
detection_rate = detected_count / total
fnr = 1 - detection_rate

if fnr > 0.05:
    issues.append(f"❌ [ADVERSARIAL] {fnr*100:.1f}% false negative rate")
    
for attack_type, metrics in adv['by_type'].items():
    rate = metrics['detected_as_malicious'] / metrics['total']
    if rate < 0.5:
        issues.append(f"  └─ {attack_type}: {rate*100:.1f}% detection rate")

# Brand bias
total = brand['total']
fps = brand.get('false_positives', [])
fp_rate = len(fps) / total
if fp_rate > 0.10:
    issues.append(f"❌ [BRAND FP] {fp_rate*100:.1f}% false positive rate on legitimate brands")

# Calibration
if calib['ece'] > 0.08:
    issues.append(f"⚠️ [CALIBRATION] ECE={calib['ece']:.4f} (poor calibration)")

if issues:
    for issue in issues:
        print(f"  {issue}")
else:
    print("  ✅ No critical issues detected!")

print("\n" + "="*80)
print("RECOMMENDATION: Train advanced 3-branch model to address these issues")
print("="*80)
