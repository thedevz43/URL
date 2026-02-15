"""
Confidence Calibration and Expected Calibration Error (ECE)
Temperature scaling and reliability diagrams for model output calibration
"""

import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class ConfidenceCalibrator:
    """Calibrate model confidence scores using temperature scaling"""
    
    def __init__(self):
        """Initialize calibrator"""
        self.temperature = 1.0
        self.is_fitted = False
    
    def compute_ece(self, y_true: np.ndarray, y_pred_probs: np.ndarray,
                    n_bins: int = 15) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE)
        
        ECE measures the difference between model confidence and accuracy
        
        Args:
            y_true: True labels (shape: [n_samples])
            y_pred_probs: Predicted probabilities (shape: [n_samples, n_classes])
            n_bins: Number of bins for calibration
        
        Returns:
            dict with ECE, MCE, bin statistics
        """
        # Get confidence (max probability) and predicted class
        confidences = np.max(y_pred_probs, axis=1)
        predictions = np.argmax(y_pred_probs, axis=1)
        accuracies = (predictions == y_true)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0  # Maximum Calibration Error
        bin_stats = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                
                # Calibration error for this bin
                bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                
                # Expected Calibration Error (weighted by bin proportion)
                ece += prop_in_bin * bin_error
                
                # Maximum Calibration Error
                mce = max(mce, bin_error)
                
                bin_stats.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'count': np.sum(in_bin),
                    'proportion': prop_in_bin,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'error': bin_error
                })
            else:
                bin_stats.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'count': 0,
                    'proportion': 0,
                    'accuracy': 0,
                    'confidence': 0,
                    'error': 0
                })
        
        return {
            'ECE': ece,
            'MCE': mce,
            'bins': bin_stats,
            'n_bins': n_bins,
            'avg_confidence': np.mean(confidences),
            'avg_accuracy': np.mean(accuracies)
        }
    
    def fit_temperature(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Find optimal temperature using validation set
        
        Temperature scaling: P_calibrated = softmax(logits / T)
        
        Args:
            logits: Model logits before softmax (shape: [n_samples, n_classes])
            y_true: True labels (shape: [n_samples])
        
        Returns:
            optimal temperature
        """
        def temperature_nll(T):
            """Negative log-likelihood with temperature scaling"""
            scaled_logits = logits / T
            # Apply softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Compute NLL
            n_samples = logits.shape[0]
            log_probs = np.log(probs[np.arange(n_samples), y_true] + 1e-12)
            nll = -np.mean(log_probs)
            
            return nll
        
        # Optimize temperature
        result = minimize(temperature_nll, x0=1.0, method='BFGS',
                         options={'disp': False})
        
        self.temperature = result.x[0]
        self.is_fitted = True
        
        return self.temperature
    
    def apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Model logits before softmax
        
        Returns:
            calibrated probabilities
        """
        if not self.is_fitted:
            print("Warning: Temperature not fitted, using T=1.0")
        
        scaled_logits = logits / self.temperature
        
        # Apply softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs
    
    def plot_reliability_diagram(self, y_true: np.ndarray, 
                                 y_pred_probs: np.ndarray,
                                 n_bins: int = 15,
                                 save_path: str = None) -> plt.Figure:
        """
        Plot reliability diagram (calibration curve)
        
        Shows relationship between predicted confidence and actual accuracy
        """
        ece_results = self.compute_ece(y_true, y_pred_probs, n_bins)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Reliability diagram
        bin_stats = ece_results['bins']
        confidences = [b['confidence'] for b in bin_stats if b['count'] > 0]
        accuracies = [b['accuracy'] for b in bin_stats if b['count'] > 0]
        counts = [b['count'] for b in bin_stats if b['count'] > 0]
        
        # Plot
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.scatter(confidences, accuracies, s=np.array(counts)/max(counts)*500,
                   alpha=0.6, c='blue', edgecolors='black', linewidths=1.5)
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title(f'Reliability Diagram\nECE = {ece_results["ECE"]:.4f}', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Confidence histogram
        all_confidences = np.max(y_pred_probs, axis=1)
        ax2.hist(all_confidences, bins=n_bins, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Confidence Distribution', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Reliability diagram saved to {save_path}")
        
        return fig
    
    def analyze_confidence_distribution(self, y_pred_probs: np.ndarray) -> Dict[str, float]:
        """
        Analyze confidence score distribution
        
        Returns statistics about model confidence
        """
        confidences = np.max(y_pred_probs, axis=1)
        
        return {
            'mean_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'q25_confidence': np.percentile(confidences, 25),
            'q75_confidence': np.percentile(confidences, 75),
            'high_confidence_rate': np.mean(confidences > 0.9),  # % predictions > 90% confidence
            'low_confidence_rate': np.mean(confidences < 0.5),   # % predictions < 50% confidence
            'very_low_confidence_rate': np.mean(confidences < 0.3),  # % predictions < 30% confidence
        }
    
    def compute_class_wise_calibration(self, y_true: np.ndarray, 
                                       y_pred_probs: np.ndarray,
                                       class_names: List[str] = None) -> Dict[str, Dict]:
        """
        Compute calibration metrics per class
        
        Useful for identifying which classes have poor calibration
        """
        n_classes = y_pred_probs.shape[1]
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        class_calibration = {}
        
        for class_idx, class_name in enumerate(class_names):
            # Binary calibration for this class
            is_class = (y_true == class_idx).astype(int)
            class_probs = y_pred_probs[:, class_idx]
            
            # Compute bins
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            
            ece = 0.0
            bin_stats = []
            
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (class_probs > bin_lower) & (class_probs <= bin_upper)
                
                if np.sum(in_bin) > 0:
                    bin_accuracy = np.mean(is_class[in_bin])
                    bin_confidence = np.mean(class_probs[in_bin])
                    bin_error = np.abs(bin_confidence - bin_accuracy)
                    
                    prop_in_bin = np.mean(in_bin)
                    ece += prop_in_bin * bin_error
                    
                    bin_stats.append({
                        'bin': i,
                        'count': np.sum(in_bin),
                        'accuracy': bin_accuracy,
                        'confidence': bin_confidence,
                        'error': bin_error
                    })
            
            class_calibration[class_name] = {
                'ECE': ece,
                'avg_probability': np.mean(class_probs),
                'class_frequency': np.mean(is_class),
                'bins': bin_stats
            }
        
        return class_calibration


class OutOfDistributionDetector:
    """Detect out-of-distribution samples using entropy and confidence"""
    
    def __init__(self, entropy_threshold: float = 1.0, 
                 confidence_threshold: float = 0.5):
        """
        Initialize OOD detector
        
        Args:
            entropy_threshold: Entropy above which sample is OOD
            confidence_threshold: Confidence below which sample is OOD
        """
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
    
    def compute_entropy(self, probs: np.ndarray) -> np.ndarray:
        """
        Compute entropy of probability distribution
        
        H = -sum(p * log(p))
        
        Higher entropy = more uncertain
        """
        # Clip to avoid log(0)
        probs_clipped = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
        
        return entropy
    
    def detect_ood(self, probs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect out-of-distribution samples
        
        Returns indices and flags for OOD samples
        """
        # Compute metrics
        entropy = self.compute_entropy(probs)
        confidence = np.max(probs, axis=1)
        
        # Flag OOD samples
        ood_by_entropy = entropy > self.entropy_threshold
        ood_by_confidence = confidence < self.confidence_threshold
        ood_combined = ood_by_entropy | ood_by_confidence
        
        return {
            'entropy': entropy,
            'confidence': confidence,
            'ood_by_entropy': ood_by_entropy,
            'ood_by_confidence': ood_by_confidence,
            'ood_combined': ood_combined,
            'ood_indices': np.where(ood_combined)[0],
            'ood_rate': np.mean(ood_combined),
            'high_entropy_rate': np.mean(ood_by_entropy),
            'low_confidence_rate': np.mean(ood_by_confidence)
        }
    
    def set_threshold_by_percentile(self, probs: np.ndarray, 
                                    percentile: float = 95) -> Tuple[float, float]:
        """
        Set thresholds based on percentile of training data
        
        Args:
            probs: Training set probabilities
            percentile: Percentile for threshold (e.g., 95 = top 5% is OOD)
        
        Returns:
            (entropy_threshold, inverse_confidence_threshold)
        """
        entropy = self.compute_entropy(probs)
        confidence = np.max(probs, axis=1)
        
        self.entropy_threshold = np.percentile(entropy, percentile)
        self.confidence_threshold = np.percentile(confidence, 100 - percentile)
        
        return self.entropy_threshold, self.confidence_threshold
    
    def analyze_uncertainty(self, probs: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive uncertainty analysis
        """
        entropy = self.compute_entropy(probs)
        confidence = np.max(probs, axis=1)
        
        # Predictive uncertainty (entropy normalized)
        max_entropy = np.log(probs.shape[1])  # log(n_classes)
        normalized_entropy = entropy / max_entropy
        
        return {
            'mean_entropy': np.mean(entropy),
            'std_entropy': np.std(entropy),
            'max_entropy': np.max(entropy),
            'mean_normalized_entropy': np.mean(normalized_entropy),
            'mean_confidence': np.mean(confidence),
            'std_confidence': np.std(confidence),
            'min_confidence': np.min(confidence),
            'uncertainty_rate_high': np.mean(normalized_entropy > 0.8),
            'uncertainty_rate_medium': np.mean((normalized_entropy > 0.5) & (normalized_entropy <= 0.8)),
            'uncertainty_rate_low': np.mean(normalized_entropy <= 0.5)
        }


if __name__ == '__main__':
    # Demo usage with synthetic data
    print("="*80)
    print("CONFIDENCE CALIBRATION DEMO")
    print("="*80)
    
    # Generate synthetic predictions
    np.random.seed(42)
    n_samples = 1000
    n_classes = 4
    
    # Simulate overconfident model
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred_probs = np.random.dirichlet([1, 1, 1, 1], n_samples)
    
    # Make predictions overconfident
    for i in range(n_samples):
        pred_class = np.argmax(y_pred_probs[i])
        y_pred_probs[i] = y_pred_probs[i] * 0.3  # Reduce other classes
        y_pred_probs[i, pred_class] = 1 - np.sum(y_pred_probs[i]) + y_pred_probs[i, pred_class]
    
    # Compute ECE
    calibrator = ConfidenceCalibrator()
    ece_results = calibrator.compute_ece(y_true, y_pred_probs)
    
    print(f"\n1. Expected Calibration Error (ECE):")
    print(f"   ECE: {ece_results['ECE']:.4f}")
    print(f"   MCE: {ece_results['MCE']:.4f}")
    print(f"   Avg Confidence: {ece_results['avg_confidence']:.4f}")
    print(f"   Avg Accuracy: {ece_results['avg_accuracy']:.4f}")
    
    # Confidence distribution
    conf_dist = calibrator.analyze_confidence_distribution(y_pred_probs)
    print(f"\n2. Confidence Distribution:")
    print(f"   Mean: {conf_dist['mean_confidence']:.4f}")
    print(f"   Median: {conf_dist['median_confidence']:.4f}")
    print(f"   High confidence (>90%): {conf_dist['high_confidence_rate']*100:.1f}%")
    print(f"   Low confidence (<50%): {conf_dist['low_confidence_rate']*100:.1f}%")
    
    # OOD detection
    print(f"\n3. Out-of-Distribution Detection:")
    ood_detector = OutOfDistributionDetector(entropy_threshold=1.0, confidence_threshold=0.5)
    ood_results = ood_detector.detect_ood(y_pred_probs)
    
    print(f"   OOD rate: {ood_results['ood_rate']*100:.1f}%")
    print(f"   High entropy: {ood_results['high_entropy_rate']*100:.1f}%")
    print(f"   Low confidence: {ood_results['low_confidence_rate']*100:.1f}%")
    
    # Uncertainty analysis
    uncertainty = ood_detector.analyze_uncertainty(y_pred_probs)
    print(f"\n4. Uncertainty Analysis:")
    print(f"   Mean entropy: {uncertainty['mean_entropy']:.4f}")
    print(f"   Mean normalized entropy: {uncertainty['mean_normalized_entropy']:.4f}")
    print(f"   High uncertainty rate: {uncertainty['uncertainty_rate_high']*100:.1f}%")
    
    print("\n" + "="*80)
