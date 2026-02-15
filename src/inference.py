"""
Production Inference Engine - v7
Enhanced inference with false positive mitigation
"""

import numpy as np
import time
import logging
from typing import Dict, Tuple, Optional
from functools import lru_cache

from src.model_loader import ModelLoader
from src.preprocess import load_preprocessor
from src.utils import DomainReputationScorer

logger = logging.getLogger(__name__)


class ProductionInferenceEngine:
    """
    Production inference engine with v7 FP mitigation
    Performance: 4% FP, 100% detection, ~47ms latency
    """
    
    VERSION = "v7_production"
    
    # Class mappings
    CLASS_NAMES = ['benign', 'defacement', 'malware', 'phishing']
    RISK_LEVELS = {
        'benign': 'low',
        'defacement': 'high',
        'malware': 'high',
        'phishing': 'high'
    }
    
    # v7 Configuration - Validated thresholds
    TIER_1_THRESHOLD = 0.93  # Very high confidence - always block
    TIER_2A_THRESHOLD = 0.75  # High confidence - reputation-based
    TIER_2B_THRESHOLD = 0.35  # Medium confidence - reputation-based
    ELITE_REPUTATION = 0.95   # Top 1000 domains
    
    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Initialize production inference engine
        
        Args:
            model_path: Path to model_v7.h5
            preprocessor_path: Path to preprocessor.pkl
        """
        self.model = ModelLoader.load_production_model(model_path)
        self.preprocessor = load_preprocessor(preprocessor_path)
        self.reputation_scorer = DomainReputationScorer()
        
        logger.info(f"Production inference engine initialized: {self.VERSION}")
    
    def predict(self, url: str, include_metadata: bool = False) -> Dict:
        """
        Predict URL classification with v7 enhanced inference
        
        Args:
            url: URL to classify
            include_metadata: Include detailed metadata in response
            
        Returns:
            Dictionary with prediction results in standardized format
        """
        start_time = time.time()
        
        try:
            # Encode URL and domain (model expects both inputs)
            url_encoded = self.preprocessor.encode_urls([url])
            domain_encoded = self.preprocessor.encode_domains([url])
            
            # Raw model prediction (multi-input)
            raw_probs = self.model.predict([url_encoded, domain_encoded], verbose=0)[0]
            
            # Calculate entropy
            entropy = self._calculate_entropy(raw_probs)
            
            # Get domain reputation
            reputation, _ = self._get_reputation(url)
            
            # Apply v7 enhanced decision logic
            prediction, confidence = self._apply_v7_logic(raw_probs, reputation)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000
            
            # Build response
            response = {
                "url": url,
                "prediction": prediction,
                "confidence": float(confidence),
                "risk_level": self.RISK_LEVELS.get(prediction, "uncertain"),
                "entropy": float(entropy),
                "inference_time_ms": float(inference_time),
                "model_version": self.VERSION
            }
            
            if include_metadata:
                response["metadata"] = {
                    "raw_probabilities": {
                        self.CLASS_NAMES[i]: float(raw_probs[i])
                        for i in range(len(self.CLASS_NAMES))
                    },
                    "domain_reputation": float(reputation),
                    "decision_logic": "v7_4tier_enhanced"
                }
            
            logger.debug(f"Prediction completed in {inference_time:.2f}ms: {prediction}")
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed for URL {url}: {str(e)}")
            return {
                "url": url,
                "prediction": "uncertain",
                "confidence": 0.0,
                "risk_level": "uncertain",
                "entropy": 0.0,
                "inference_time_ms": 0.0,
                "model_version": self.VERSION,
                "error": str(e)
            }
    
    def _apply_v7_logic(self, raw_probs: np.ndarray, reputation: float) -> Tuple[str, float]:
        """
        Apply v7 4-tier enhanced decision logic
        
        Args:
            raw_probs: Raw model probabilities [benign, defacement, malware, phishing]
            reputation: Domain reputation score (0-1)
            
        Returns:
            (prediction_class, confidence_score)
        """
        benign_prob = raw_probs[0]
        defacement_prob = raw_probs[1]
        malware_prob = raw_probs[2]
        phishing_prob = raw_probs[3]
        
        # Get max malicious probability
        max_malicious_prob = max(defacement_prob, malware_prob, phishing_prob)
        
        # TIER 1: Very High Confidence (>=93%) - ALWAYS BLOCK
        if max_malicious_prob >= self.TIER_1_THRESHOLD:
            if malware_prob > phishing_prob:
                return 'malware', float(malware_prob)
            else:
                return 'phishing', float(phishing_prob)
        
        # TIER 2A: High Confidence (75-93%) - REPUTATION-BASED
        elif max_malicious_prob >= self.TIER_2A_THRESHOLD:
            if reputation >= self.ELITE_REPUTATION:
                # Elite domain protection
                return 'benign', float(benign_prob)
            else:
                # Block all others
                if malware_prob > phishing_prob:
                    return 'malware', float(malware_prob)
                else:
                    return 'phishing', float(phishing_prob)
        
        # TIER 2B: Medium Confidence (35-75%) - STRICT REPUTATION CHECK
        elif max_malicious_prob >= self.TIER_2B_THRESHOLD:
            if reputation >= self.ELITE_REPUTATION:
                # Only elite domains allowed
                return 'benign', float(benign_prob)
            else:
                # Block all others
                if malware_prob > phishing_prob:
                    return 'malware', float(malware_prob)
                else:
                    return 'phishing', float(phishing_prob)
        
        # TIER 3: Low Confidence (<35%) - BENIGN
        else:
            return 'benign', float(benign_prob)
    
    @staticmethod
    def _calculate_entropy(probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy of probability distribution"""
        probs = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)
    
    @lru_cache(maxsize=10000)
    def _get_reputation(self, url: str) -> Tuple[float, float]:
        """Get cached domain reputation"""
        domain = self.reputation_scorer.extract_domain(url)
        if not domain:
            return 0.0, 0.5
        return self.reputation_scorer.get_reputation_score('http://' + domain), 0.0
