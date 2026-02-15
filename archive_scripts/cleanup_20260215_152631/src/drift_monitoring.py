"""
Model Drift Monitoring System

Tracks model performance degradation over time by monitoring:
1. Prediction confidence distribution shifts
2. Class frequency changes
3. Feature distribution drift
4. Entropy patterns
5. Performance metrics

Provides alerts when significant drift is detected to trigger retraining.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt


class DriftDetector:
    """
    Monitors model predictions for distribution drift
    
    Uses multiple statistical tests to detect when model behavior
    changes significantly from baseline, indicating need for retraining
    """
    
    def __init__(
        self,
        window_size=1000,
        confidence_threshold=0.15,
        entropy_threshold=0.2,
        frequency_threshold=0.1,
        psi_threshold=0.2
    ):
        """
        Initialize drift detector
        
        Args:
            window_size: Number of recent predictions to track
            confidence_threshold: Alert if mean confidence drops by this much
            entropy_threshold: Alert if entropy increases by this much
            frequency_threshold: Alert if class frequency changes by this much
            psi_threshold: Population Stability Index threshold for features
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.frequency_threshold = frequency_threshold
        self.psi_threshold = psi_threshold
        
        # Sliding windows for metrics
        self.confidence_window = deque(maxlen=window_size)
        self.entropy_window = deque(maxlen=window_size)
        self.prediction_window = deque(maxlen=window_size)
        self.feature_window = deque(maxlen=window_size)
        
        # Baseline statistics (set during calibration)
        self.baseline_confidence_mean = None
        self.baseline_confidence_std = None
        self.baseline_entropy_mean = None
        self.baseline_entropy_std = None
        self.baseline_class_dist = None
        self.baseline_feature_dist = None
        
        # Alert history
        self.alerts = []
    
    # ========================================================================
    # BASELINE CALIBRATION
    # ========================================================================
    
    def calibrate_baseline(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        features: Optional[np.ndarray] = None,
        entropies: Optional[np.ndarray] = None
    ):
        """
        Set baseline distribution from validation set
        
        Args:
            confidences: Max confidence scores (N,)
            predictions: Predicted classes (N,)
            features: Feature matrix (N, n_features), optional
            entropies: Prediction entropy values (N,), optional
        """
        print("="*80)
        print("CALIBRATING DRIFT DETECTION BASELINE")
        print("="*80)
        print()
        
        # Confidence baseline
        self.baseline_confidence_mean = np.mean(confidences)
        self.baseline_confidence_std = np.std(confidences)
        print(f"Confidence baseline: μ={self.baseline_confidence_mean:.4f}, σ={self.baseline_confidence_std:.4f}")
        
        # Entropy baseline
        if entropies is not None:
            self.baseline_entropy_mean = np.mean(entropies)
            self.baseline_entropy_std = np.std(entropies)
            print(f"Entropy baseline: μ={self.baseline_entropy_mean:.4f}, σ={self.baseline_entropy_std:.4f}")
        else:
            # Compute entropy from predictions if not provided
            unique_classes = np.unique(predictions)
            class_probs = np.array([np.mean(predictions == c) for c in unique_classes])
            self.baseline_entropy_mean = -np.sum(class_probs * np.log(class_probs + 1e-10))
            self.baseline_entropy_std = 0.1  # Default
            print(f"Entropy baseline (computed): μ={self.baseline_entropy_mean:.4f}")
        
        # Class distribution baseline
        unique, counts = np.unique(predictions, return_counts=True)
        self.baseline_class_dist = dict(zip(unique.astype(int), counts / len(predictions)))
        print()
        print("Baseline class distribution:")
        for cls, freq in sorted(self.baseline_class_dist.items()):
            print(f"  Class {cls}: {freq:.4f}")
        
        # Feature distribution baseline (for PSI)
        if features is not None:
            self.baseline_feature_dist = self._compute_feature_distribution(features)
            print(f"\nFeature distribution baseline: {features.shape[1]} features tracked")
        
        print()
        print("✓ Baseline calibration complete")
        print("="*80)
        print()
    
    def _compute_feature_distribution(self, features: np.ndarray) -> List[Dict]:
        """
        Compute feature distribution for PSI calculation
        
        Returns list of dicts with histogram info for each feature
        """
        n_bins = 10
        feature_dists = []
        
        for i in range(features.shape[1]):
            feat = features[:, i]
            hist, bin_edges = np.histogram(feat, bins=n_bins)
            hist = hist / len(feat)  # Normalize to probabilities
            
            feature_dists.append({
                'hist': hist,
                'bin_edges': bin_edges
            })
        
        return feature_dists
    
    # ========================================================================
    # DRIFT DETECTION
    # ========================================================================
    
    def update(
        self,
        confidence: float,
        prediction: int,
        features: Optional[np.ndarray] = None,
        entropy: Optional[float] = None
    ):
        """
        Add new prediction to monitoring window
        
        Args:
            confidence: Max confidence score
            prediction: Predicted class
            features: Feature vector (n_features,), optional
            entropy: Prediction entropy, optional
        """
        self.confidence_window.append(confidence)
        self.prediction_window.append(prediction)
        
        if entropy is not None:
            self.entropy_window.append(entropy)
        
        if features is not None:
            self.feature_window.append(features)
    
    def detect_drift(self) -> Dict:
        """
        Check for distribution drift
        
        Returns:
            Dict with drift detection results and alert flags
        """
        if len(self.confidence_window) < 100:
            return {
                'drift_detected': False,
                'reason': 'Insufficient samples',
                'n_samples': len(self.confidence_window)
            }
        
        alerts = []
        drift_scores = {}
        
        # 1. Confidence drift
        conf_drift, conf_score = self._detect_confidence_drift()
        if conf_drift:
            alerts.append('Confidence drop detected')
        drift_scores['confidence'] = conf_score
        
        # 2. Entropy drift
        if len(self.entropy_window) >= 100:
            entropy_drift, entropy_score = self._detect_entropy_drift()
            if entropy_drift:
                alerts.append('Entropy increase detected')
            drift_scores['entropy'] = entropy_score
        
        # 3. Class frequency drift
        freq_drift, freq_score = self._detect_frequency_drift()
        if freq_drift:
            alerts.append('Class distribution shift detected')
        drift_scores['class_frequency'] = freq_score
        
        # 4. Feature drift (PSI)
        if len(self.feature_window) >= 100 and self.baseline_feature_dist is not None:
            psi_drift, psi_score = self._detect_feature_drift()
            if psi_drift:
                alerts.append('Feature distribution drift detected')
            drift_scores['feature_psi'] = psi_score
        
        # Record alert if any drift detected
        drift_detected = len(alerts) > 0
        if drift_detected:
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'alerts': alerts,
                'drift_scores': drift_scores,
                'window_size': len(self.confidence_window)
            }
            self.alerts.append(alert_record)
        
        return {
            'drift_detected': drift_detected,
            'alerts': alerts,
            'drift_scores': drift_scores,
            'n_samples': len(self.confidence_window),
            'baseline_stats': {
                'confidence_mean': self.baseline_confidence_mean,
                'entropy_mean': self.baseline_entropy_mean,
                'class_dist': self.baseline_class_dist
            },
            'current_stats': {
                'confidence_mean': np.mean(list(self.confidence_window)),
                'entropy_mean': np.mean(list(self.entropy_window)) if self.entropy_window else None,
                'class_dist': self._get_current_class_dist()
            }
        }
    
    def _detect_confidence_drift(self) -> Tuple[bool, float]:
        """Detect if confidence has dropped significantly"""
        current_mean = np.mean(list(self.confidence_window))
        
        # z-score relative to baseline
        z_score = (self.baseline_confidence_mean - current_mean) / (self.baseline_confidence_std + 1e-10)
        
        # Alert if confidence dropped by threshold amount
        drop = self.baseline_confidence_mean - current_mean
        drift_detected = drop > self.confidence_threshold
        
        return drift_detected, float(drop)
    
    def _detect_entropy_drift(self) -> Tuple[bool, float]:
        """Detect if entropy has increased significantly"""
        current_mean = np.mean(list(self.entropy_window))
        
        # Alert if entropy increased (more uncertainty)
        increase = current_mean - self.baseline_entropy_mean
        drift_detected = increase > self.entropy_threshold
        
        return drift_detected, float(increase)
    
    def _detect_frequency_drift(self) -> Tuple[bool, float]:
        """Detect if class prediction frequencies have shifted"""
        current_dist = self._get_current_class_dist()
        
        # Chi-square test
        expected_counts = []
        observed_counts = []
        
        all_classes = set(self.baseline_class_dist.keys()) | set(current_dist.keys())
        
        for cls in all_classes:
            baseline_freq = self.baseline_class_dist.get(cls, 0)
            current_freq = current_dist.get(cls, 0)
            
            expected_counts.append(baseline_freq * len(self.prediction_window))
            observed_counts.append(current_freq * len(self.prediction_window))
        
        # Chi-square statistic
        expected = np.array(expected_counts) + 1  # Add smoothing
        observed = np.array(observed_counts) + 1
        chi2 = np.sum((observed - expected) ** 2 / expected)
        
        # Max frequency difference
        max_diff = max([abs(current_dist.get(cls, 0) - self.baseline_class_dist.get(cls, 0))
                       for cls in all_classes])
        
        drift_detected = max_diff > self.frequency_threshold
        
        return drift_detected, float(max_diff)
    
    def _detect_feature_drift(self) -> Tuple[bool, float]:
        """Detect feature distribution drift using PSI"""
        recent_features = np.array(list(self.feature_window))
        
        # Compute PSI for each feature
        psi_scores = []
        
        for i, baseline_dist in enumerate(self.baseline_feature_dist):
            feat = recent_features[:, i]
            
            # Compute current distribution using same bins
            current_hist, _ = np.histogram(feat, bins=baseline_dist['bin_edges'])
            current_hist = current_hist / len(feat)  # Normalize
            
            # PSI = sum((current - baseline) * log(current / baseline))
            baseline_hist = baseline_dist['hist']
            
            # Add small constant to avoid log(0)
            current_hist = current_hist + 1e-10
            baseline_hist = baseline_hist + 1e-10
            
            psi = np.sum((current_hist - baseline_hist) * np.log(current_hist / baseline_hist))
            psi_scores.append(abs(psi))
        
        # Use max PSI across features
        max_psi = max(psi_scores)
        drift_detected = max_psi > self.psi_threshold
        
        return drift_detected, float(max_psi)
    
    def _get_current_class_dist(self) -> Dict:
        """Get current class distribution from window"""
        predictions = np.array(list(self.prediction_window))
        unique, counts = np.unique(predictions, return_counts=True)
        return dict(zip(unique.astype(int), counts / len(predictions)))
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def get_alert_history(self) -> List[Dict]:
        """Return all recorded alerts"""
        return self.alerts
    
    def generate_drift_report(self, output_path: str = 'drift_report.json'):
        """
        Generate comprehensive drift monitoring report
        
        Args:
            output_path: Path to save JSON report
        """
        if not self.alerts:
            report = {
                'summary': 'No drift detected',
                'total_alerts': 0,
                'monitoring_window': self.window_size,
                'current_samples': len(self.confidence_window)
            }
        else:
            report = {
                'summary': f'{len(self.alerts)} drift alert(s) recorded',
                'total_alerts': len(self.alerts),
                'monitoring_window': self.window_size,
                'current_samples': len(self.confidence_window),
                'alert_history': self.alerts,
                'baseline_stats': {
                    'confidence': {
                        'mean': float(self.baseline_confidence_mean),
                        'std': float(self.baseline_confidence_std)
                    },
                    'entropy': {
                        'mean': float(self.baseline_entropy_mean),
                        'std': float(self.baseline_entropy_std)
                    },
                    'class_distribution': {int(k): float(v) for k, v in self.baseline_class_dist.items()}
                },
                'current_stats': {
                    'confidence': {
                        'mean': float(np.mean(list(self.confidence_window))),
                        'std': float(np.std(list(self.confidence_window)))
                    },
                    'entropy': {
                        'mean': float(np.mean(list(self.entropy_window))) if self.entropy_window else None,
                        'std': float(np.std(list(self.entropy_window))) if self.entropy_window else None
                    },
                    'class_distribution': {int(k): float(v) for k, v in self._get_current_class_dist().items()}
                },
                'recommendations': self._get_recommendations()
            }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Drift report saved to {output_path}")
        
        return report
    
    def _get_recommendations(self) -> List[str]:
        """Generate recommendations based on drift patterns"""
        recommendations = []
        
        if not self.alerts:
            return ['Continue monitoring. No action needed.']
        
        # Check recent alerts
        recent_alerts = self.alerts[-3:] if len(self.alerts) >= 3 else self.alerts
        
        alert_types = set()
        for alert in recent_alerts:
            alert_types.update(alert['alerts'])
        
        if 'Confidence drop detected' in alert_types:
            recommendations.append(
                'CRITICAL: Model confidence has dropped significantly. '
                'Review recent predictions and consider retraining.'
            )
        
        if 'Entropy increase detected' in alert_types:
            recommendations.append(
                'WARNING: Model uncertainty has increased. '
                'Data distribution may have shifted. Investigate recent inputs.'
            )
        
        if 'Class distribution shift detected' in alert_types:
            recommendations.append(
                'ALERT: Class prediction frequencies have changed. '
                'Potential concept drift or sampling bias.'
            )
        
        if 'Feature distribution drift detected' in alert_types:
            recommendations.append(
                'NOTICE: Input feature distributions have drifted. '
                'Check data pipeline and preprocessing.'
            )
        
        # General recommendations
        if len(self.alerts) >= 3:
            recommendations.append(
                'Multiple drift alerts detected. Recommend model retraining with recent data.'
            )
        
        return recommendations
    
    def plot_drift_trends(self, save_path: Optional[str] = None):
        """
        Plot drift monitoring trends
        
        Args:
            save_path: Path to save figure, or None to display
        """
        if len(self.confidence_window) < 100:
            print("Insufficient data for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Drift Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Confidence over time
        ax = axes[0, 0]
        confidence_vals = list(self.confidence_window)
        ax.plot(confidence_vals, alpha=0.6, linewidth=0.5)
        ax.axhline(self.baseline_confidence_mean, color='green', linestyle='--', 
                   label=f'Baseline: {self.baseline_confidence_mean:.3f}')
        ax.axhline(self.baseline_confidence_mean - self.confidence_threshold, 
                   color='red', linestyle=':', label='Alert Threshold')
        ax.set_title('Confidence Trend')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Max Confidence')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Entropy over time
        ax = axes[0, 1]
        if len(self.entropy_window) >= 100:
            entropy_vals = list(self.entropy_window)
            ax.plot(entropy_vals, alpha=0.6, linewidth=0.5, color='orange')
            ax.axhline(self.baseline_entropy_mean, color='green', linestyle='--',
                       label=f'Baseline: {self.baseline_entropy_mean:.3f}')
            ax.axhline(self.baseline_entropy_mean + self.entropy_threshold,
                       color='red', linestyle=':', label='Alert Threshold')
            ax.set_title('Entropy Trend')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Prediction Entropy')
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient entropy data', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 3. Class distribution comparison
        ax = axes[1, 0]
        current_dist = self._get_current_class_dist()
        classes = sorted(set(self.baseline_class_dist.keys()) | set(current_dist.keys()))
        
        baseline_freqs = [self.baseline_class_dist.get(c, 0) for c in classes]
        current_freqs = [current_dist.get(c, 0) for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        ax.bar(x - width/2, baseline_freqs, width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, current_freqs, width, label='Current', alpha=0.8)
        ax.set_title('Class Distribution Comparison')
        ax.set_xlabel('Class')
        ax.set_ylabel('Frequency')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # 4. Alert timeline
        ax = axes[1, 1]
        if self.alerts:
            alert_times = list(range(len(self.alerts)))
            alert_counts = [len(alert['alerts']) for alert in self.alerts]
            ax.bar(alert_times, alert_counts, alpha=0.7, color='red')
            ax.set_title('Alert Timeline')
            ax.set_xlabel('Alert Event')
            ax.set_ylabel('Number of Alerts')
            ax.grid(alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No alerts recorded', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Drift trends plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    """Test drift detection"""
    
    print("Testing Drift Detection System")
    print()
    
    # Simulate baseline data
    np.random.seed(42)
    n_baseline = 5000
    
    # Baseline: high confidence, low entropy, balanced classes
    baseline_confidences = np.random.beta(8, 2, n_baseline)  # Mean ~0.8
    baseline_predictions = np.random.choice([0, 1, 2, 3], n_baseline, 
                                           p=[0.7, 0.15, 0.10, 0.05])
    baseline_features = np.random.randn(n_baseline, 10)
    baseline_entropies = -baseline_confidences * np.log(baseline_confidences)
    
    # Initialize detector
    detector = DriftDetector(window_size=1000)
    
    # Calibrate baseline
    detector.calibrate_baseline(
        confidences=baseline_confidences,
        predictions=baseline_predictions,
        features=baseline_features,
        entropies=baseline_entropies
    )
    
    # Simulate normal operation (no drift)
    print("\nPhase 1: Normal operation (no drift)")
    for i in range(500):
        conf = np.random.beta(8, 2)
        pred = np.random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.10, 0.05])
        feat = np.random.randn(10)
        entropy = -conf * np.log(conf)
        
        detector.update(conf, pred, feat, entropy)
    
    result = detector.detect_drift()
    print(f"Drift detected: {result['drift_detected']}")
    if result['drift_detected']:
        print(f"Alerts: {result['alerts']}")
    
    # Simulate drift: confidence drop
    print("\nPhase 2: Simulating confidence drop")
    for i in range(500):
        conf = np.random.beta(4, 4)  # Lower confidence (mean ~0.5)
        pred = np.random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.10, 0.05])
        feat = np.random.randn(10)
        entropy = -conf * np.log(conf + 1e-10)
        
        detector.update(conf, pred, feat, entropy)
    
    result = detector.detect_drift()
    print(f"Drift detected: {result['drift_detected']}")
    if result['drift_detected']:
        print(f"Alerts: {result['alerts']}")
        print(f"Drift scores: {result['drift_scores']}")
    
    # Generate report
    print("\nGenerating drift report...")
    report = detector.generate_drift_report('test_drift_report.json')
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
