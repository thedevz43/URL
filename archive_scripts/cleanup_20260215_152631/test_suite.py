"""
Comprehensive Stress-Testing Framework for Malicious URL Detection Model

This module implements rigorous testing for model robustness including:
1. Temporal testing (old data → new data simulation)
2. Adversarial URL generation (phishing, homoglyphs, subdomain chains)
3. Brand bias testing (top-ranked domains)
4. Robustness tests (edge cases, extreme inputs)
5. Confidence calibration (ECE, temperature scaling)
6. Out-of-distribution detection

"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tensorflow import keras
from src.model import focal_loss
from src.preprocess import load_and_preprocess_data
from src.adversarial_generators import AdversarialURLGenerator, BrandDomainGenerator
from src.robustness_tests import RobustnessTestGenerator, test_model_robustness
from src.calibration import ConfidenceCalibrator, OutOfDistributionDetector


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class ComprehensiveStressTest:
    """Main stress testing framework"""
    
    def __init__(self, model_path: str, preprocessor_path: str, 
                 data_path: str = 'data/malicious_phish.csv'):
        """
        Initialize stress testing framework
        
        Args:
            model_path: Path to trained model (.h5)
            preprocessor_path: Path to preprocessor (.pkl)
            data_path: Path to dataset for temporal testing
        """
        print("="*80)
        print("COMPREHENSIVE STRESS-TESTING FRAMEWORK")
        print("="*80)
        print()
        
        # Load model
        print("[1] Loading model and preprocessor...")
        self.model = keras.models.load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss()}
        )
        print(f"✓ Model loaded from {model_path}")
        
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        print(f"✓ Preprocessor loaded from {preprocessor_path}")
        
        self.data_path = data_path
        self.class_names = list(self.preprocessor.label_encoder.classes_)
        
        # Initialize generators
        self.adv_generator = AdversarialURLGenerator(seed=42)
        self.brand_generator = BrandDomainGenerator()
        self.robust_generator = RobustnessTestGenerator(seed=42)
        
        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'tests': {}
        }
        
        print()
    
    def run_temporal_test(self) -> Dict[str, Any]:
        """
        Simulate temporal degradation: train on old data, test on new data
        
        Uses stratified split to simulate temporal drift:
        - 'Old' data: First 60% (training simulation)
        - 'New' data: Last 40% (test simulation)
        """
        print("="*80)
        print("TEST 1: TEMPORAL DEGRADATION ANALYSIS")
        print("="*80)
        print()
        
        print("[1/3] Loading full dataset...")
        df = pd.read_csv(self.data_path)
        
        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates()
        print(f"  Loaded {initial_size} URLs ({len(df)} after deduplication)")
        
        # Simulate temporal split (assume older data is earlier in dataset)
        split_idx = int(len(df) * 0.6)
        old_data = df.iloc[:split_idx]
        new_data = df.iloc[split_idx:]
        
        print(f"\n[2/3] Simulating temporal split...")
        print(f"  'Old' training data: {len(old_data)} URLs")
        print(f"  'New' test data: {len(new_data)} URLs")
        
        # Evaluate on both splits
        print(f"\n[3/3] Evaluating model performance...")
        
        old_results = self._evaluate_on_data(old_data, "Old Data (training period)")
        new_results = self._evaluate_on_data(new_data, "New Data (future period)")
        
        # Calculate degradation
        degradation = {
            'accuracy_old': old_results['accuracy'],
            'accuracy_new': new_results['accuracy'],
            'accuracy_drop': old_results['accuracy'] - new_results['accuracy'],
            'accuracy_drop_percent': ((old_results['accuracy'] - new_results['accuracy']) / 
                                     old_results['accuracy'] * 100),
            'class_wise_degradation': {}
        }
        
        for class_name in self.class_names:
            old_acc = old_results['class_accuracy'][class_name]
            new_acc = new_results['class_accuracy'][class_name]
            degradation['class_wise_degradation'][class_name] = {
                'old_accuracy': old_acc,
                'new_accuracy': new_acc,
                'drop': old_acc - new_acc
            }
        
        print(f"\n{'Results:':-^80}")
        print(f"Old Data Accuracy: {old_results['accuracy']:.4f}")
        print(f"New Data Accuracy: {new_results['accuracy']:.4f}")
        print(f"Performance Drop: {degradation['accuracy_drop']:.4f} ({degradation['accuracy_drop_percent']:.2f}%)")
        
        if degradation['accuracy_drop'] > 0.02:
            print(f"⚠️ WARNING: Significant temporal degradation detected!")
        else:
            print(f"✓ Model shows good temporal stability")
        
        print()
        
        return {
            'old_results': old_results,
            'new_results': new_results,
            'degradation': degradation
        }
    
    def run_adversarial_test(self, n_samples: int = 200) -> Dict[str, Any]:
        """
        Test model on adversarial URLs (should detect as malicious)
        
        Includes: homoglyphs, subdomain chains, typosquatting, etc.
        """
        print("="*80)
        print("TEST 2: ADVERSARIAL URL GENERATION")
        print("="*80)
        print()
        
        print(f"[1/2] Generating {n_samples} adversarial URLs...")
        adversarial_urls = self.adv_generator.generate_adversarial_batch(n_samples)
        
        # Count by type
        type_counts = {}
        for url_data in adversarial_urls:
            t = url_data['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"  Generated adversarial URLs by type:")
        for t, count in sorted(type_counts.items()):
            print(f"    {t:25} : {count:3} URLs")
        
        print(f"\n[2/2] Evaluating model on adversarial URLs...")
        
        # Predict on adversarial URLs
        results = {
            'total': len(adversarial_urls),
            'by_type': {},
            'false_negatives': [],  # Adversarial URLs classified as benign
            'predictions': []
        }
        
        for url_data in adversarial_urls:
            url = url_data['url']
            url_type = url_data['type']
            
            try:
                # Encode and predict
                url_seq = self.preprocessor.encode_urls([url])
                domain_seq = self.preprocessor.encode_domains([url])
                pred_probs = self.model.predict([url_seq, domain_seq], verbose=0)[0]
                pred_class = int(np.argmax(pred_probs))
                pred_label = self.class_names[pred_class]
                
                # Track prediction
                prediction = {
                    'url': url[:100],
                    'type': url_type,
                    'predicted_class': pred_label,
                    'confidence': float(pred_probs[pred_class]),
                    'all_probs': pred_probs.tolist()
                }
                
                results['predictions'].append(prediction)
                
                # Count by type
                if url_type not in results['by_type']:
                    results['by_type'][url_type] = {
                        'total': 0,
                        'detected_as_malicious': 0,
                        'detected_as_benign': 0,
                        'predictions': []
                    }
                
                results['by_type'][url_type]['total'] += 1
                results['by_type'][url_type]['predictions'].append(pred_label)
                
                # Check if detected as malicious (phishing, malware, or defacement)
                if pred_label in ['phishing', 'malware', 'defacement']:
                    results['by_type'][url_type]['detected_as_malicious'] += 1
                else:
                    results['by_type'][url_type]['detected_as_benign'] += 1
                    results['false_negatives'].append(prediction)
            
            except Exception as e:
                print(f"  Error processing {url[:50]}: {e}")
        
        # Calculate detection rates
        total_malicious_detected = sum(r['detected_as_malicious'] for r in results['by_type'].values())
        total_benign_misclass = sum(r['detected_as_benign'] for r in results['by_type'].values())
        
        detection_rate = total_malicious_detected / results['total'] if results['total'] > 0 else 0
        false_negative_rate = total_benign_misclass / results['total'] if results['total'] > 0 else 0
        
        results['overall_detection_rate'] = detection_rate
        results['false_negative_rate'] = false_negative_rate
        
        print(f"\n{'Results:':-^80}")
        print(f"Overall Detection Rate: {detection_rate:.2%}")
        print(f"False Negative Rate: {false_negative_rate:.2%} ({total_benign_misclass}/{results['total']} missed)")
        
        print(f"\nDetection by attack type:")
        for attack_type, stats in sorted(results['by_type'].items()):
            det_rate = stats['detected_as_malicious'] / stats['total'] if stats['total'] > 0 else 0
            status = "✓" if det_rate > 0.8 else "⚠️" if det_rate > 0.5 else "❌"
            print(f"  {status} {attack_type:25} : {det_rate:6.1%} ({stats['detected_as_malicious']}/{stats['total']})")
        
        if false_negative_rate > 0.1:
            print(f"\n⚠️ WARNING: High false negative rate on adversarial URLs!")
        else:
            print(f"\n✓ Good adversarial detection performance")
        
        print()
        
        return results
    
    def run_brand_bias_test(self, n_samples: int = 50) -> Dict[str, Any]:
        """
        Test model on legitimate top-ranked domains
        Should have very low false positive rate
        """
        print("="*80)
        print("TEST 3: BRAND BIAS / FALSE POSITIVE ANALYSIS")
        print("="*80)
        print()
        
        print(f"[1/2] Generating {n_samples} legitimate brand URLs...")
        brand_urls = self.brand_generator.generate_legitimate_urls(n_samples)
        
        print(f"  Generated URLs from top domains")
        print(f"  Examples: {', '.join([u['domain'] for u in brand_urls[:5]])}, ...")
        
        print(f"\n[2/2] Testing for false positives...")
        
        results = {
            'total': len(brand_urls),
            'correct_benign': 0,
            'false_positives': [],
            'predictions': []
        }
        
        for url_data in brand_urls:
            url = url_data['url']
            domain = url_data['domain']
            
            try:
                # Predict
                url_seq = self.preprocessor.encode_urls([url])
                domain_seq = self.preprocessor.encode_domains([url])
                pred_probs = self.model.predict([url_seq, domain_seq], verbose=0)[0]
                pred_class = int(np.argmax(pred_probs))
                pred_label = self.class_names[pred_class]
                
                prediction = {
                    'url': url,
                    'domain': domain,
                    'predicted_class': pred_label,
                    'confidence': float(pred_probs[pred_class]),
                    'benign_prob': float(pred_probs[0])  # Assuming benign is class 0
                }
                
                results['predictions'].append(prediction)
                
                if pred_label == 'benign':
                    results['correct_benign'] += 1
                else:
                    results['false_positives'].append(prediction)
            
            except Exception as e:
                print(f"  Error processing {url}: {e}")
        
        # Calculate false positive rate
        fp_rate = len(results['false_positives']) / results['total'] if results['total'] > 0 else 0
        accuracy = results['correct_benign'] / results['total'] if results['total'] > 0 else 0
        
        results['false_positive_rate'] = fp_rate
        results['accuracy'] = accuracy
        
        print(f"\n{'Results:':-^80}")
        print(f"Legitimate URLs tested: {results['total']}")
        print(f"Correctly classified as benign: {results['correct_benign']} ({accuracy:.2%})")
        print(f"False Positive Rate: {fp_rate:.2%}")
        
        if results['false_positives']:
            print(f"\nFalse Positives (legitimate sites misclassified):")
            for i, fp in enumerate(results['false_positives'][:10], 1):
                print(f"  {i}. {fp['domain']:30} → {fp['predicted_class']:12} ({fp['confidence']:.2%})")
        
        if fp_rate > 0.05:
            print(f"\n⚠️ WARNING: High false positive rate on legitimate brands!")
        else:
            print(f"\n✓ Low false positive rate - good for production")
        
        print()
        
        return results
    
    def run_robustness_test(self) -> Dict[str, Any]:
        """
        Test model on edge cases: empty strings, extreme lengths, noise, etc.
        Model should never crash
        """
        print("="*80)
        print("TEST 4: ROBUSTNESS & EDGE CASE HANDLING")
        print("="*80)
        print()
        
        print("[1/2] Generating robustness test cases...")
        test_cases = self.robust_generator.generate_complete_robustness_suite()
        
        print(f"  Generated {len(test_cases)} edge case tests")
        
        # Count by type
        type_counts = {}
        for case in test_cases:
            t = case['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"\n  Test cases by type:")
        for t, count in sorted(type_counts.items()):
            print(f"    {t:25} : {count:3} cases")
        
        print(f"\n[2/2] Testing model robustness (should never crash)...")
        
        results = test_model_robustness(self.model, self.preprocessor, test_cases)
        
        print(f"\n{'Results:':-^80}")
        print(f"Total tests: {results['total_tests']}")
        print(f"Successful: {results['successful']} ({results['success_rate']:.2%})")
        print(f"Errors/Crashes: {results['errors']}")
        
        if results['errors'] > 0:
            print(f"\n⚠️ Crashes detected ({results['errors']} cases):")
            for crash in results['crashes'][:10]:  # Show first 10
                print(f"  • Type: {crash['type']:20} Error: {crash['error_type']:15} - {crash['error'][:50]}")
        else:
            print(f"\n✓ Model handled all edge cases without crashing!")
        
        # Analyze error patterns
        if results['errors'] > 0:
            error_types = {}
            for crash in results['crashes']:
                et = crash['error_type']
                error_types[et] = error_types.get(et, 0) + 1
            
            print(f"\nError breakdown:")
            for et, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {et:20} : {count:3} occurrences")
        
        print()
        
        return results
    
    def run_calibration_analysis(self, n_samples: int = 10000) -> Dict[str, Any]:
        """
        Analyze confidence calibration using test set
        Compute ECE, plot reliability diagrams
        """
        print("="*80)
        print("TEST 5: CONFIDENCE CALIBRATION ANALYSIS")
        print("="*80)
        print()
        
        print("[1/3] Loading test data for calibration...")
        # Load data
        X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, preprocessor = \
            load_and_preprocess_data(self.data_path, test_size=0.2, random_state=42, multi_input=True)
        
        # Sample if too large
        if len(y_test) > n_samples:
            indices = np.random.choice(len(y_test), n_samples, replace=False)
            X_test = [X_test[0][indices], X_test[1][indices]]
            y_test = y_test[indices]
            y_test_cat = y_test_cat[indices]
        
        print(f"  Using {len(y_test)} test samples")
        
        print(f"\n[2/3] Computing predictions and calibration metrics...")
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test, batch_size=256, verbose=0)
        
        # Compute ECE
        calibrator = ConfidenceCalibrator()
        ece_results = calibrator.compute_ece(y_test, y_pred_probs)
        
        # Confidence distribution
        conf_dist = calibrator.analyze_confidence_distribution(y_pred_probs)
        
        # Class-wise calibration
        class_calibration = calibrator.compute_class_wise_calibration(
            y_test, y_pred_probs, self.class_names
        )
        
        print(f"\n[3/3] Plotting reliability diagram...")
        calibrator.plot_reliability_diagram(
            y_test, y_pred_probs, n_bins=15,
            save_path='models/stress_test_calibration.png'
        )
        
        results = {
            'ece': ece_results['ECE'],
            'mce': ece_results['MCE'],
            'avg_confidence': ece_results['avg_confidence'],
            'avg_accuracy': ece_results['avg_accuracy'],
            'confidence_distribution': conf_dist,
            'class_wise_calibration': class_calibration,
            'bin_statistics': ece_results['bins']
        }
        
        print(f"\n{'Results:':-^80}")
        print(f"Expected Calibration Error (ECE): {ece_results['ECE']:.4f}")
        print(f"Maximum Calibration Error (MCE): {ece_results['MCE']:.4f}")
        print(f"Average Confidence: {ece_results['avg_confidence']:.4f}")
        print(f"Average Accuracy: {ece_results['avg_accuracy']:.4f}")
        
        print(f"\nConfidence Distribution:")
        print(f"  High confidence (>90%): {conf_dist['high_confidence_rate']:.2%}")
        print(f"  Low confidence (<50%): {conf_dist['low_confidence_rate']:.2%}")
        print(f"  Very low (<30%): {conf_dist['very_low_confidence_rate']:.2%}")
        
        print(f"\nClass-wise ECE:")
        for class_name, stats in class_calibration.items():
            print(f"  {class_name:12} : ECE = {stats['ECE']:.4f}")
        
        if ece_results['ECE'] < 0.05:
            print(f"\n✓ Excellent calibration (ECE < 0.05)")
        elif ece_results['ECE'] < 0.10:
            print(f"\n✓ Good calibration (ECE < 0.10)")
        else:
            print(f"\n⚠️ Poor calibration - model confidence unreliable")
        
        print()
        
        return results
    
    def run_ood_detection(self, n_samples: int = 5000) -> Dict[str, Any]:
        """
        Test out-of-distribution detection using entropy and confidence
        """
        print("="*80)
        print("TEST 6: OUT-OF-DISTRIBUTION DETECTION")
        print("="*80)
        print()
        
        print("[1/3] Loading test data...")
        X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, preprocessor = \
            load_and_preprocess_data(self.data_path, test_size=0.2, random_state=42, multi_input=True)
        
        # Sample
        if len(y_test) > n_samples:
            indices = np.random.choice(len(y_test), n_samples, replace=False)
            X_test = [X_test[0][indices], X_test[1][indices]]
            y_test = y_test[indices]
        
        print(f"  Using {len(y_test)} test samples")
        
        print(f"\n[2/3] Computing predictions and entropy...")
        y_pred_probs = self.model.predict(X_test, batch_size=256, verbose=0)
        
        # Initialize OOD detector
        ood_detector = OutOfDistributionDetector(
            entropy_threshold=1.0,
            confidence_threshold=0.5
        )
        
        # Detect OOD
        ood_results = ood_detector.detect_ood(y_pred_probs)
        
        # Uncertainty analysis
        uncertainty = ood_detector.analyze_uncertainty(y_pred_probs)
        
        print(f"\n[3/3] Generating adversarial URLs for OOD testing...")
        # Test on adversarial URLs (should have higher entropy/lower confidence)
        adv_urls = self.adv_generator.generate_adversarial_batch(100)
        adv_probs = []
        
        for url_data in adv_urls[:50]:  # Test subset
            try:
                url = url_data['url']
                url_seq = self.preprocessor.encode_urls([url])
                domain_seq = self.preprocessor.encode_domains([url])
                pred_probs = self.model.predict([url_seq, domain_seq], verbose=0)
                adv_probs.append(pred_probs[0])
            except:
                pass
        
        if adv_probs:
            adv_probs = np.array(adv_probs)
            adv_ood = ood_detector.detect_ood(adv_probs)
            adv_uncertainty = ood_detector.analyze_uncertainty(adv_probs)
        else:
            adv_ood = None
            adv_uncertainty = None
        
        results = {
            'test_set_ood': ood_results,
            'test_set_uncertainty': uncertainty,
            'adversarial_ood': adv_ood,
            'adversarial_uncertainty': adv_uncertainty
        }
        
        print(f"\n{'Results (Test Set):':-^80}")
        print(f"OOD Rate (combined): {ood_results['ood_rate']:.2%}")
        print(f"High Entropy Rate: {ood_results['high_entropy_rate']:.2%}")
        print(f"Low Confidence Rate: {ood_results['low_confidence_rate']:.2%}")
        print(f"\nMean Entropy: {uncertainty['mean_entropy']:.4f}")
        print(f"Mean Confidence: {uncertainty['mean_confidence']:.4f}")
        
        if adv_ood:
            print(f"\n{'Results (Adversarial URLs):':-^80}")
            print(f"OOD Rate: {adv_ood['ood_rate']:.2%}")
            print(f"Mean Entropy: {adv_uncertainty['mean_entropy']:.4f}")
            print(f"Mean Confidence: {adv_uncertainty['mean_confidence']:.4f}")
        
        print()
        
        return results
    
    def _evaluate_on_data(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """Helper to evaluate model on a dataframe"""
        urls = df['url'].values
        labels = df['type'].values
        
        # Encode
        url_seqs = self.preprocessor.encode_urls(urls)
        domain_seqs = self.preprocessor.encode_domains(urls)
        label_encoded = self.preprocessor.label_encoder.transform(labels)
        
        # Predict
        pred_probs = self.model.predict([url_seqs, domain_seqs], batch_size=256, verbose=0)
        pred_classes = np.argmax(pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(pred_classes == label_encoded)
        
        # Per-class accuracy
        class_accuracy = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = (label_encoded == i)
            if np.sum(class_mask) > 0:
                class_acc = np.mean(pred_classes[class_mask] == label_encoded[class_mask])
                class_accuracy[class_name] = class_acc
            else:
                class_accuracy[class_name] = 0.0
        
        return {
            'name': name,
            'n_samples': len(urls),
            'accuracy': accuracy,
            'class_accuracy': class_accuracy
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete stress test suite"""
        print("="*80)
        print("RUNNING COMPLETE STRESS TEST SUITE")
        print("="*80)
        print()
        
        # Run all tests
        try:
            self.results['tests']['temporal'] = self.run_temporal_test()
        except Exception as e:
            print(f"❌ Temporal test failed: {e}\n")
            self.results['tests']['temporal'] = {'error': str(e)}
        
        try:
            self.results['tests']['adversarial'] = self.run_adversarial_test()
        except Exception as e:
            print(f"❌ Adversarial test failed: {e}\n")
            self.results['tests']['adversarial'] = {'error': str(e)}
        
        try:
            self.results['tests']['brand_bias'] = self.run_brand_bias_test()
        except Exception as e:
            print(f"❌ Brand bias test failed: {e}\n")
            self.results['tests']['brand_bias'] = {'error': str(e)}
        
        try:
            self.results['tests']['robustness'] = self.run_robustness_test()
        except Exception as e:
            print(f"❌ Robustness test failed: {e}\n")
            self.results['tests']['robustness'] = {'error': str(e)}
        
        try:
            self.results['tests']['calibration'] = self.run_calibration_analysis()
        except Exception as e:
            print(f"❌ Calibration test failed: {e}\n")
            self.results['tests']['calibration'] = {'error': str(e)}
        
        try:
            self.results['tests']['ood_detection'] = self.run_ood_detection()
        except Exception as e:
            print(f"❌ OOD detection test failed: {e}\n")
            self.results['tests']['ood_detection'] = {'error': str(e)}
        
        return self.results
    
    def generate_report(self, save_path: str = 'models/stress_test_report.json'):
        """Generate comprehensive test report"""
        # Convert numpy types to native Python types
        serializable_results = convert_numpy_types(self.results)
        
        # Save JSON report
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✓ Detailed report saved to {save_path}")
        
        # Generate summary
        self._print_summary()
    
    def _print_summary(self):
        """Print executive summary of all tests"""
        print("\n")
        print("="*80)
        print("STRESS TEST EXECUTIVE SUMMARY")
        print("="*80)
        print()
        
        tests = self.results['tests']
        
        # Overall assessment
        critical_issues = []
        warnings = []
        passes = []
        
        # Temporal
        if 'temporal' in tests and 'error' not in tests['temporal']:
            deg = tests['temporal']['degradation']
            if deg['accuracy_drop'] > 0.05:
                critical_issues.append(f"Temporal degradation: {deg['accuracy_drop']:.2%} drop")
            elif deg['accuracy_drop'] > 0.02:
                warnings.append(f"Moderate temporal degradation: {deg['accuracy_drop']:.2%}")
            else:
                passes.append("Temporal stability: ✓")
        
        # Adversarial
        if 'adversarial' in tests and 'error' not in tests['adversarial']:
            fnr = tests['adversarial']['false_negative_rate']
            if fnr > 0.2:
                critical_issues.append(f"High false negative rate on adversarial: {fnr:.2%}")
            elif fnr > 0.1:
                warnings.append(f"Moderate adversarial vulnerability: {fnr:.2%} FNR")
            else:
                passes.append(f"Adversarial robustness: ✓ ({(1-fnr):.1%} detection)")
        
        # Brand bias
        if 'brand_bias' in tests and 'error' not in tests['brand_bias']:
            fpr = tests['brand_bias']['false_positive_rate']
            if fpr > 0.10:
                critical_issues.append(f"High false positive rate on brands: {fpr:.2%}")
            elif fpr > 0.05:
                warnings.append(f"Elevated brand false positives: {fpr:.2%}")
            else:
                passes.append(f"Brand classification: ✓ ({fpr:.2%} FPR)")
        
        # Robustness
        if 'robustness' in tests and 'error' not in tests['robustness']:
            success_rate = tests['robustness']['success_rate']
            if success_rate < 0.8:
                critical_issues.append(f"Model crashes on edge cases: {success_rate:.1%} success")
            elif success_rate < 0.95:
                warnings.append(f"Some edge case failures: {success_rate:.1%} success")
            else:
                passes.append(f"Edge case handling: ✓ ({success_rate:.1%} success)")
        
        # Calibration
        if 'calibration' in tests and 'error' not in tests['calibration']:
            ece = tests['calibration']['ece']
            if ece > 0.15:
                critical_issues.append(f"Poor confidence calibration: ECE={ece:.4f}")
            elif ece > 0.10:
                warnings.append(f"Moderate calibration issues: ECE={ece:.4f}")
            else:
                passes.append(f"Confidence calibration: ✓ (ECE={ece:.4f})")
        
        # Print summary
        print("CRITICAL ISSUES:")
        if critical_issues:
            for issue in critical_issues:
                print(f"  ❌ {issue}")
        else:
            print("  ✓ None")
        
        print("\nWARNINGS:")
        if warnings:
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        else:
            print("  ✓ None")
        
        print("\nPASSED TESTS:")
        for p in passes:
            print(f"  {p}")
        
        # Overall verdict
        print("\n" + "="*80)
        if critical_issues:
            print("OVERALL VERDICT: ❌ CRITICAL ISSUES DETECTED - NOT PRODUCTION READY")
        elif warnings:
            print("OVERALL VERDICT: ⚠️  WARNINGS PRESENT - REVIEW BEFORE DEPLOYMENT")
        else:
            print("OVERALL VERDICT: ✓ ALL TESTS PASSED - PRODUCTION READY")
        print("="*80)
        print()


def main():
    """Main entry point for stress testing"""
    # Configuration
    MODEL_PATH = 'models/url_detector_improved.h5'
    PREPROCESSOR_PATH = 'models/preprocessor_improved.pkl'
    DATA_PATH = 'data/malicious_phish.csv'
    
    # Initialize and run tests
    tester = ComprehensiveStressTest(MODEL_PATH, PREPROCESSOR_PATH, DATA_PATH)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate report
    tester.generate_report()
    
    print("\n" + "="*80)
    print("STRESS TESTING COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  • models/stress_test_report.json - Detailed results")
    print("  • models/stress_test_calibration.png - Calibration plots")
    print("\nNext steps:")
    print("  1. Review stress_test_report.json for detailed findings")
    print("  2. Address any critical issues or warnings")
    print("  3. Re-run tests after improvements")
    print()


if __name__ == '__main__':
    main()
