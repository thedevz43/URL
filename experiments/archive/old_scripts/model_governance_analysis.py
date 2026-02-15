"""
ML Model Governance Analysis
Evaluates all trained models and selects the best for production
"""

import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

class ModelGovernanceAnalyzer:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        
    def discover_models(self):
        """Discover all model files and their metadata"""
        model_files = list(self.models_dir.glob('url_detector*.h5'))
        
        for model_file in model_files:
            model_name = model_file.stem
            print(f"\n📦 Discovering model: {model_name}")
            
            # Determine metadata file
            if 'advanced' in model_name:
                metadata_file = 'training_metadata_advanced.json'
                preprocessor = 'preprocessor_advanced.pkl'
            elif 'augmented' in model_name:
                metadata_file = 'training_metadata_augmented.json'
                preprocessor = 'preprocessor_augmented.pkl'
            elif 'improved'  in model_name:
                metadata_file = 'training_metadata_improved.json'
                preprocessor = 'preprocessor_improved.pkl'
            else:
                metadata_file = 'training_metadata.json'
                preprocessor = 'preprocessor.pkl'
            
            self.models[model_name] = {
                'model_file': str(model_file),
                'metadata_file': str(self.models_dir / metadata_file),
                'preprocessor_file': str(self.models_dir / preprocessor),
                'model_path': model_file
            }
            
            # Load metadata
            try:
                with open(self.models_dir / metadata_file, 'r') as f:
                    self.models[model_name]['metadata'] = json.load(f)
                print(f"  ✓ Loaded metadata")
            except Exception as e:
                print(f"  ✗ Failed to load metadata: {e}")
                self.models[model_name]['metadata'] = {}
        
        # Load stress test report (shared by all models trained after)
        stress_test_file = self.models_dir / 'stress_test_report.json'
        if stress_test_file.exists():
            with open(stress_test_file, 'r') as f:
                stress_data = json.load(f)
                # Determine which model this belongs to
                model_path = stress_data.get('model_path', '')
                for name, data in self.models.items():
                    if name in model_path or model_path.endswith(f'{name}.h5'):
                        self.models[name]['stress_test'] = stress_data
                        print(f"  ✓ Loaded stress test for {name}")
                        break
        
        # Load evaluation metrics
        eval_file = self.models_dir / 'evaluation_metrics_detailed.json'
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
                # This is for the most recently evaluated model
                # Based on the stress test, this is for url_detector_improved.h5
                if 'url_detector_improved' in self.models:
                    self.models['url_detector_improved']['evaluation'] = eval_data
                    print(f"  ✓ Loaded evaluation metrics for url_detector_improved")
        
        # Load comprehensive test results
        comp_test_file = self.models_dir / 'comprehensive_test_results.json'
        if comp_test_file.exists():
            with open(comp_test_file, 'r') as f:
                comp_data = json.load(f)
                model_path = comp_data.get('model_path', '')
                for name, data in self.models.items():
                    if name in model_path:
                        self.models[name]['comprehensive_test'] = comp_data
                        print(f"  ✓ Loaded comprehensive test for {name}")
                        break
        
        # Load brand test results
        brand_test_file = self.models_dir / 'detailed_brand_test_results.json'
        if brand_test_file.exists():
            with open(brand_test_file, 'r') as f:
                brand_data = json.load(f)
                # Associate with url_detector_improved (based on enhanced inference tests)
                if 'url_detector_improved' in self.models:
                    self.models['url_detector_improved']['brand_test'] = brand_data
                    print(f"  ✓ Loaded brand test results for url_detector_improved")
        
        return len(self.models)
    
    def evaluate_model(self, model_name):
        """Evaluate a single model against governance criteria"""
        data = self.models[model_name]
        metadata = data.get('metadata', {})
        stress_test = data.get('stress_test', {})
        evaluation = data.get('evaluation', {})
        comp_test = data.get('comprehensive_test', {})
        brand_test = data.get('brand_test', {})
        
        result = {
            'model_name': model_name,
            'training_date': metadata.get('training_date', metadata.get('timestamp', 'Unknown')),
            'parameters': metadata.get('model_parameters', metadata.get('total_parameters', 0)),
        }
        
        # Extract performance metrics
        if evaluation:
            result['test_accuracy'] = evaluation.get('test_accuracy', 0)
            result['test_loss'] = evaluation.get('test_loss', 0)
            
            # Get class-wise metrics
            report = evaluation.get('classification_report', {})
            result['benign_recall'] = report.get('benign', {}).get('recall', 0)
            result['phishing_recall'] = report.get('phishing', {}).get('recall', 0)
            result['malware_recall'] = report.get('malware', {}).get('recall', 0)
            result['defacement_recall'] = report.get('defacement', {}).get('recall', 0)
            
            # Calculate false positive rate (1 - benign recall = benign classified as malicious)
            # Actually, FP rate = FP / (FP + TN)
            # For benign class: FP = samples incorrectly classified as malicious
            # Let me use precision/recall to estimate
            benign_precision = report.get('benign', {}).get('precision', 0)
            benign_recall = report.get('benign', {}).get('recall', 0)
            
            # FP rate for legitimate domains = 1 - benign_recall (assuming benign is legitimate)
            result['fp_rate'] = (1 - benign_recall) * 100
            
            # Detection rate = weighted average of malicious class recalls
            phishing_support = report.get('phishing', {}).get('support', 0)
            malware_support = report.get('malware', {}).get('support', 0)
            defacement_support = report.get('defacement', {}).get('support', 0)
            total_malicious = phishing_support + malware_support + defacement_support
            
            if total_malicious > 0:
                detection_rate = (
                    (result['phishing_recall'] * phishing_support +
                     result['malware_recall'] * malware_support +
                     result['defacement_recall'] * defacement_support) / total_malicious
                )
                result['detection_rate'] = detection_rate * 100
            else:
                result['detection_rate'] = 0
        else:
            # Use metadata performance
            perf = metadata.get('performance', {})
            result['test_accuracy'] = perf.get('test_accuracy', metadata.get('final_val_accuracy', 0))
            result['test_loss'] = perf.get('test_loss', metadata.get('final_val_loss', 0))
            result['fp_rate'] = None  # Unknown
            result['detection_rate'] = None  # Unknown
        
        # Stress test results
        if stress_test and 'tests' in stress_test:
            tests = stress_test['tests']
            
            # Temporal stability
            if 'temporal' in tests:
                temporal = tests['temporal']
                degradation = temporal.get('degradation', {})
                result['temporal_accuracy_drop'] = degradation.get('accuracy_drop_percent', 0)
                result['temporal_stability'] = 100 - result['temporal_accuracy_drop']
            else:
                result['temporal_accuracy_drop'] = None
                result['temporal_stability'] = None
            
            # Adversarial robustness
            if 'adversarial' in tests:
                adv = tests['adversarial']
                total = adv.get('total', 0)
                fn_count = len(adv.get('false_negatives', []))
                if total > 0:
                    result['adversarial_detection_rate'] = ((total - fn_count) / total) * 100
                else:
                    result['adversarial_detection_rate'] = None
            else:
                result['adversarial_detection_rate'] = None
            
            # Brand FP test
            if 'brand_fp' in tests:
                brand = tests['brand_fp']
                result['brand_fp_count'] = brand.get('false_positives', 0)
                result['brand_total'] = brand.get('total_brands', 0)
                if result['brand_total'] > 0:
                    result['brand_fp_rate'] = (result['brand_fp_count'] / result['brand_total']) * 100
                else:
                    result['brand_fp_rate'] = None
            else:
                result['brand_fp_count'] = None
                result['brand_fp_rate'] = None
        else:
            result['temporal_accuracy_drop'] = None
            result['temporal_stability'] = None
            result['adversarial_detection_rate'] = None
            result['brand_fp_count'] = None
            result['brand_fp_rate'] = None
        
        # Estimated inference latency (based on parameters)
        # Rough estimate: ~0.001ms per 1000 parameters
        result['estimated_latency_ms'] = (result['parameters'] / 1000) * 0.1
        
        return result
    
    def check_mandatory_criteria(self, result):
        """Check if model meets mandatory governance criteria"""
        failures = []
        
        # FP Rate ≤ 5%
        if result['fp_rate'] is not None:
            if result['fp_rate'] > 5.0:
                failures.append(f"FP Rate {result['fp_rate']:.2f}% > 5%")
        else:
            # If we have brand FP rate, use that
            if result['brand_fp_rate'] is not None:
                if result['brand_fp_rate'] > 5.0:
                    failures.append(f"Brand FP Rate {result['brand_fp_rate']:.2f}% > 5%")
            else:
                failures.append("FP Rate unknown")
        
        # Detection Rate ≥ 95%
        if result['detection_rate'] is not None:
            if result['detection_rate'] < 95.0:
                failures.append(f"Detection Rate {result['detection_rate']:.2f}% < 95%")
        else:
            failures.append("Detection Rate unknown")
        
        # No critical stress test failures (temporal drop > 10%)
        if result['temporal_accuracy_drop'] is not None:
            if result['temporal_accuracy_drop'] > 10.0:
                failures.append(f"Temporal degradation {result['temporal_accuracy_drop']:.2f}% > 10%")
        
        result['mandatory_failures'] = failures
        result['passes_mandatory'] = len(failures) == 0
        
        return result
    
    def calculate_score(self, result):
        """Calculate overall score for ranking"""
        score = 0
        
        # Security metrics (60% weight)
        if result['detection_rate'] is not None:
            score += (result['detection_rate'] / 100) * 40  # 40 points max
        
        if result['fp_rate'] is not None:
            # Lower FP is better: max 20 points at 0% FP, 0 points at 10% FP
            fp_score = max(0, 20 * (1 - result['fp_rate'] / 10))
            score += fp_score
        elif result['brand_fp_rate'] is not None:
            fp_score = max(0, 20 * (1 - result['brand_fp_rate'] / 10))
            score += fp_score
        
        # Reliability metrics (25% weight)
        if result['temporal_stability'] is not None:
            score += (result['temporal_stability'] / 100) * 15  # 15 points max
        
        if result['adversarial_detection_rate'] is not None:
            score += (result['adversarial_detection_rate'] / 100) * 10  # 10 points max
        
        # Performance metrics (15% weight)
        # Lower latency is better: max 15 points at 10ms, 0 points at 100ms
        if result['estimated_latency_ms'] > 0:
            latency_score = max(0, 15 * (1 - (result['estimated_latency_ms'] - 10) / 90))
            score += latency_score
        
        result['governance_score'] = round(score, 2)
        return result
    
    def rank_models(self):
        """Rank all models by governance criteria"""
        results = []
        
        for model_name in self.models:
            result = self.evaluate_model(model_name)
            result = self.check_mandatory_criteria(result)
            result = self.calculate_score(result)
            results.append(result)
        
        # Sort by: passes_mandatory DESC, governance_score DESC
        results.sort(key=lambda x: (x['passes_mandatory'], x['governance_score']), reverse=True)
        
        return results
    
    def generate_report(self, results):
        """Generate comparison report"""
        print("\n" + "="*100)
        print("MODEL GOVERNANCE ANALYSIS - PRODUCTION SELECTION")
        print("="*100)
        
        print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Models Evaluated: {len(results)}")
        
        # Summary table
        print("\n" + "-"*100)
        print("RANKING TABLE")
        print("-"*100)
        
        df = pd.DataFrame(results)
        
        # Select columns for display
        display_cols = [
            'model_name', 'passes_mandatory', 'governance_score',
            'fp_rate', 'detection_rate', 'temporal_stability',
            'test_accuracy', 'estimated_latency_ms'
        ]
        
        display_df = df[[col for col in display_cols if col in df.columns]].copy()
        
        # Format for display
        if 'fp_rate' in display_df.columns:
            display_df['fp_rate'] = display_df['fp_rate'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
        if 'detection_rate' in display_df.columns:
            display_df['detection_rate'] = display_df['detection_rate'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
        if 'temporal_stability' in display_df.columns:
            display_df['temporal_stability'] = display_df['temporal_stability'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
        if 'test_accuracy' in display_df.columns:
            display_df['test_accuracy'] = display_df['test_accuracy'].apply(lambda x: f"{x:.4f}")
        if 'estimated_latency_ms' in display_df.columns:
            display_df['estimated_latency_ms'] = display_df['estimated_latency_ms'].apply(lambda x: f"{x:.2f}ms")
        
        print(display_df.to_string(index=False))
        
        # Detailed analysis
        print("\n" + "-"*100)
        print("DETAILED EVALUATION")
        print("-"*100)
        
        for i, result in enumerate(results, 1):
            print(f"\n#{i} {result['model_name']}")
            print(f"  Governance Score: {result['governance_score']:.2f}/100")
            print(f"  Training Date: {result['training_date']}")
            print(f"  Parameters: {result['parameters']:,}")
            
            print(f"\n  MANDATORY CRITERIA:")
            if result['passes_mandatory']:
                print(f"    ✅ PASSES ALL MANDATORY REQUIREMENTS")
            else:
                print(f"    ❌ FAILS MANDATORY REQUIREMENTS:")
                for failure in result['mandatory_failures']:
                    print(f"       - {failure}")
            
            print(f"\n  SECURITY METRICS:")
            if result['fp_rate'] is not None:
                status = "✅" if result['fp_rate'] <= 5.0 else "❌"
                print(f"    {status} False Positive Rate: {result['fp_rate']:.2f}% (target ≤5%)")
            elif result['brand_fp_rate'] is not None:
                status = "✅" if result['brand_fp_rate'] <= 5.0 else "❌"
                print(f"    {status} Brand FP Rate: {result['brand_fp_rate']:.2f}% (target ≤5%)")
            else:
                print(f"    ⚠️  FP Rate: Not available")
            
            if result['detection_rate'] is not None:
                status = "✅" if result['detection_rate'] >= 95.0 else "❌"
                print(f"    {status} Detection Rate: {result['detection_rate']:.2f}% (target ≥95%)")
            else:
                print(f"    ⚠️  Detection Rate: Not available")
            
            print(f"\n  RELIABILITY METRICS:")
            if result['temporal_stability'] is not None:
                status = "✅" if result['temporal_accuracy_drop'] <= 10.0 else "❌"
                print(f"    {status} Temporal Stability: {result['temporal_stability']:.2f}% (decay: {result['temporal_accuracy_drop']:.2f}%)")
            else:
                print(f"    ⚠️  Temporal Stability: Not tested")
            
            if result['adversarial_detection_rate'] is not None:
                print(f"    Adversarial Detection: {result['adversarial_detection_rate']:.2f}%")
            
            print(f"\n  PERFORMANCE:")
            print(f"    Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"    Estimated Latency: {result['estimated_latency_ms']:.2f}ms")
        
        # Selection
        print("\n" + "="*100)
        print("PRODUCTION MODEL SELECTION")
        print("="*100)
        
        selected = results[0]
        
        print(f"\n🏆 SELECTED MODEL: {selected['model_name']}")
        print(f"   Governance Score: {selected['governance_score']:.2f}/100")
        print(f"   Mandatory Criteria: {'✅ PASS' if selected['passes_mandatory'] else '❌ FAIL'}")
        
        if not selected['passes_mandatory']:
            print(f"\n⚠️  WARNING: Selected model does not meet all mandatory criteria!")
            print(f"   Recommend further optimization before production deployment.")
        
        return selected


def main():
    analyzer = ModelGovernanceAnalyzer()
    
    print("="*100)
    print("ML MODEL GOVERNANCE - PRODUCTION MODEL SELECTION")
    print("="*100)
    
    # Discover models
    n_models = analyzer.discover_models()
    print(f"\n✓ Discovered {n_models} models")
    
    # Rank models
    print("\n📊 Evaluating models against governance criteria...")
    results = analyzer.rank_models()
    
    # Generate report
    selected = analyzer.generate_report(results)
    
    # Save report
    report_file = 'model_governance_report.json'
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': len(results),
            'selected_model': selected['model_name'],
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Detailed report saved to: {report_file}")
    
    return selected


if __name__ == "__main__":
    selected_model = main()
    print(f"\n{'='*100}")
    print(f"Next Step: Run restructure_models.py to promote {selected_model['model_name']} to production")
    print(f"{'='*100}")
