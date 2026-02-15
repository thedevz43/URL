"""
Advanced Feature Engineering for URL Security Analysis

Extracts handcrafted numeric features that complement deep learning:
- Length-based features
- Character composition (digits, special chars)
- Entropy measurements
- Suspicious keyword detection
- TLD risk scoring
- Domain reputation signals

These features address limitations that character-level CNNs struggle with:
1. Global URL properties
2. Statistical anomalies
3. Known malicious patterns
4. Domain reputation
"""

import numpy as np
import re
from urllib.parse import urlparse
from collections import Counter
import math
from typing import List, Dict, Tuple


class URLFeatureExtractor:
    """Extract handcrafted security-relevant features from URLs"""
    
    # Suspicious keywords commonly found in malicious URLs
    SUSPICIOUS_KEYWORDS = [
        'login', 'signin', 'account', 'verify', 'secure', 'update', 'confirm',
        'banking', 'paypal', 'amazon', 'apple', 'microsoft', 'facebook',
        'password', 'credit', 'card', 'ssn', 'validation', 'suspended',
        'unusual', 'activity', 'click', 'here', 'free', 'winner', 'prize',
        'urgent', 'expire', 'limited', 'offer', 'reset', 'unlock', 'restore',
        'admin', 'administrator', 'root', 'wp-admin', 'phpmyadmin',
        'script', 'cmd', 'exec', 'eval', 'shell', 'backdoor'
    ]
    
    # High-risk TLDs (frequently abused in phishing/malware)
    HIGH_RISK_TLDS = {
        'tk': 5.0, 'ml': 5.0, 'ga': 5.0, 'cf': 5.0, 'gq': 5.0,  # Free domains
        'xyz': 3.0, 'top': 3.0, 'work': 3.0, 'click': 4.0, 'link': 3.5,
        'pw': 3.0, 'cc': 2.5, 'club': 2.5, 'download': 4.0, 'racing': 3.0,
        'stream': 3.0, 'trade': 3.0, 'accountant': 3.5, 'science': 3.0,
        'loan': 3.5, 'faith': 3.0, 'win': 3.5, 'bid': 3.0, 'date': 3.0
    }
    
    # Moderate-risk TLDs
    MODERATE_RISK_TLDS = {
        'info': 1.5, 'biz': 1.5, 'online': 1.5, 'site': 1.5, 'website': 1.5,
        'space': 1.5, 'fun': 1.5, 'tech': 1.0, 'store': 1.0
    }
    
    # Trusted TLDs (less likely to be malicious)
    TRUSTED_TLDS = {
        'com': 0.1, 'org': 0.2, 'net': 0.3, 'edu': 0.05, 'gov': 0.01,
        'mil': 0.01, 'int': 0.05, 'uk': 0.2, 'de': 0.2, 'fr': 0.2,
        'jp': 0.2, 'ca': 0.2, 'au': 0.2, 'cn': 0.4
    }
    
    def __init__(self):
        """Initialize feature extractor"""
        self.feature_names = [
            'url_length',
            'domain_length',
            'path_length',
            'num_subdomains',
            'digit_ratio',
            'special_char_ratio',
            'uppercase_ratio',
            'url_entropy',
            'domain_entropy',
            'suspicious_keyword_count',
            'tld_risk_score',
            'has_ip_address',
            'num_dots',
            'num_hyphens',
            'num_underscores',
            'longest_word_length',
            'num_query_params',
            'has_port',
            'is_https',
            'num_slashes'
        ]
        
        self.n_features = len(self.feature_names)
    
    def extract_features(self, url: str) -> np.ndarray:
        """
        Extract all features from a single URL
        
        Args:
            url: URL string to analyze
            
        Returns:
            numpy array of shape (n_features,)
        """
        try:
            parsed = urlparse(url)
            
            features = [
                self._url_length(url),
                self._domain_length(parsed),
                self._path_length(parsed),
                self._num_subdomains(parsed),
                self._digit_ratio(url),
                self._special_char_ratio(url),
                self._uppercase_ratio(url),
                self._url_entropy(url),
                self._domain_entropy(parsed),
                self._suspicious_keyword_count(url),
                self._tld_risk_score(parsed),
                self._has_ip_address(parsed),
                self._num_dots(url),
                self._num_hyphens(url),
                self._num_underscores(url),
                self._longest_word_length(url),
                self._num_query_params(parsed),
                self._has_port(parsed),
                self._is_https(parsed),
                self._num_slashes(url)
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            # Return zeros if parsing fails
            return np.zeros(self.n_features, dtype=np.float32)
    
    def extract_batch(self, urls: List[str]) -> np.ndarray:
        """
        Extract features from multiple URLs
        
        Args:
            urls: List of URL strings
            
        Returns:
            numpy array of shape (n_urls, n_features)
        """
        features = [self.extract_features(url) for url in urls]
        return np.array(features, dtype=np.float32)
    
    # ============================================================================
    # FEATURE EXTRACTION METHODS
    # ============================================================================
    
    def _url_length(self, url: str) -> float:
        """Total URL length (normalized)"""
        return min(len(url) / 200.0, 1.0)
    
    def _domain_length(self, parsed) -> float:
        """Domain length (normalized)"""
        domain = parsed.netloc
        return min(len(domain) / 50.0, 1.0)
    
    def _path_length(self, parsed) -> float:
        """Path length (normalized)"""
        path = parsed.path
        return min(len(path) / 100.0, 1.0)
    
    def _num_subdomains(self, parsed) -> float:
        """Number of subdomains (normalized)"""
        domain = parsed.netloc.split(':')[0]  # Remove port
        parts = domain.split('.')
        # Subtract 2 for domain.tld
        num_subs = max(len(parts) - 2, 0)
        return min(num_subs / 5.0, 1.0)
    
    def _digit_ratio(self, url: str) -> float:
        """Ratio of digits to total characters"""
        if len(url) == 0:
            return 0.0
        digits = sum(c.isdigit() for c in url)
        return digits / len(url)
    
    def _special_char_ratio(self, url: str) -> float:
        """Ratio of special characters to total"""
        if len(url) == 0:
            return 0.0
        special = sum(not c.isalnum() for c in url)
        return special / len(url)
    
    def _uppercase_ratio(self, url: str) -> float:
        """Ratio of uppercase letters"""
        if len(url) == 0:
            return 0.0
        letters = sum(c.isalpha() for c in url)
        if letters == 0:
            return 0.0
        uppercase = sum(c.isupper() for c in url)
        return uppercase / letters
    
    def _url_entropy(self, url: str) -> float:
        """Shannon entropy of URL (normalized)"""
        return self._calculate_entropy(url) / 8.0  # Normalize by max entropy
    
    def _domain_entropy(self, parsed) -> float:
        """Shannon entropy of domain (normalized)"""
        domain = parsed.netloc.split(':')[0]
        return self._calculate_entropy(domain) / 8.0
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy"""
        if len(text) == 0:
            return 0.0
        
        counter = Counter(text)
        length = len(text)
        entropy = 0.0
        
        for count in counter.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _suspicious_keyword_count(self, url: str) -> float:
        """Count of suspicious keywords (normalized)"""
        url_lower = url.lower()
        count = sum(keyword in url_lower for keyword in self.SUSPICIOUS_KEYWORDS)
        return min(count / 5.0, 1.0)  # Normalize by 5
    
    def _tld_risk_score(self, parsed) -> float:
        """TLD-based risk score (0-1)"""
        domain = parsed.netloc.split(':')[0]
        
        if not domain:
            return 0.5
        
        # Extract TLD
        parts = domain.split('.')
        if len(parts) < 2:
            return 0.5
        
        tld = parts[-1].lower()
        
        # Check risk categories
        if tld in self.HIGH_RISK_TLDS:
            return min(self.HIGH_RISK_TLDS[tld] / 5.0, 1.0)
        elif tld in self.MODERATE_RISK_TLDS:
            return min(self.MODERATE_RISK_TLDS[tld] / 5.0, 1.0)
        elif tld in self.TRUSTED_TLDS:
            return self.TRUSTED_TLDS[tld]
        else:
            return 0.3  # Unknown TLD - moderate risk
    
    def _has_ip_address(self, parsed) -> float:
        """Binary: domain is IP address"""
        domain = parsed.netloc.split(':')[0]
        
        # Check IPv4
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ipv4_pattern, domain):
            return 1.0
        
        # Check IPv6 (simplified)
        if ':' in domain and domain.count(':') >= 2:
            return 1.0
        
        return 0.0
    
    def _num_dots(self, url: str) -> float:
        """Number of dots (normalized)"""
        return min(url.count('.') / 10.0, 1.0)
    
    def _num_hyphens(self, url: str) -> float:
        """Number of hyphens (normalized)"""
        return min(url.count('-') / 10.0, 1.0)
    
    def _num_underscores(self, url: str) -> float:
        """Number of underscores (normalized)"""
        return min(url.count('_') / 5.0, 1.0)
    
    def _longest_word_length(self, url: str) -> float:
        """Longest continuous alphanumeric sequence (normalized)"""
        words = re.findall(r'[a-zA-Z0-9]+', url)
        if not words:
            return 0.0
        max_len = max(len(word) for word in words)
        return min(max_len / 50.0, 1.0)
    
    def _num_query_params(self, parsed) -> float:
        """Number of query parameters (normalized)"""
        query = parsed.query
        if not query:
            return 0.0
        params = query.split('&')
        return min(len(params) / 10.0, 1.0)
    
    def _has_port(self, parsed) -> float:
        """Binary: non-standard port specified"""
        port = parsed.port
        if port is None:
            return 0.0
        # Standard ports
        if port in [80, 443]:
            return 0.0
        return 1.0
    
    def _is_https(self, parsed) -> float:
        """Binary: uses HTTPS"""
        return 1.0 if parsed.scheme == 'https' else 0.0
    
    def _num_slashes(self, url: str) -> float:
        """Number of slashes (normalized)"""
        return min(url.count('/') / 10.0, 1.0)
    
    def get_feature_importance_report(self, features: np.ndarray, 
                                     predictions: np.ndarray) -> Dict:
        """
        Analyze which features are most predictive
        
        Args:
            features: Feature matrix (n_samples, n_features)
            predictions: Binary predictions (0=benign, 1=malicious)
            
        Returns:
            Dictionary with feature statistics
        """
        report = {}
        
        benign_mask = predictions == 0
        malicious_mask = predictions == 1
        
        for i, name in enumerate(self.feature_names):
            benign_values = features[benign_mask, i]
            malicious_values = features[malicious_mask, i]
            
            report[name] = {
                'benign_mean': float(np.mean(benign_values)) if len(benign_values) > 0 else 0.0,
                'malicious_mean': float(np.mean(malicious_values)) if len(malicious_values) > 0 else 0.0,
                'benign_std': float(np.std(benign_values)) if len(benign_values) > 0 else 0.0,
                'malicious_std': float(np.std(malicious_values)) if len(malicious_values) > 0 else 0.0,
                'difference': float(np.mean(malicious_values) - np.mean(benign_values)) if len(benign_values) > 0 and len(malicious_values) > 0 else 0.0
            }
        
        return report


if __name__ == "__main__":
    """Test feature extraction"""
    
    extractor = URLFeatureExtractor()
    
    # Test URLs
    test_urls = [
        'https://github.com/user/repo',
        'http://paypal-verify-account-urgent.tk/login.php?id=12345',
        'https://192.168.1.1:8080/admin/config.php',
        'https://docs.python.org/3/library/urllib.parse.html',
        'http://free-prize-winner-click-here-now.xyz/claim?ref=email&utm_source=spam'
    ]
    
    print("="*80)
    print("URL FEATURE EXTRACTION TEST")
    print("="*80)
    print()
    
    for url in test_urls:
        print(f"URL: {url[:60]}...")
        features = extractor.extract_features(url)
        
        print("\nTop features:")
        # Show top 10 highest-value features
        top_indices = np.argsort(features)[-10:][::-1]
        for idx in top_indices:
            if features[idx] > 0.1:  # Only show significant features
                print(f"  {extractor.feature_names[idx]:25} : {features[idx]:.3f}")
        print()
        print("-"*80)
        print()
    
    # Batch extraction
    print(f"\nBatch extraction shape: {extractor.extract_batch(test_urls).shape}")
    print(f"Expected: ({len(test_urls)}, {extractor.n_features})")
