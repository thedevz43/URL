"""
Preprocessing utilities for URL malicious detection
Production version - v7
"""

import numpy as np
import pickle
from typing import List, Union
from tensorflow.keras.preprocessing.sequence import pad_sequences


class URLPreprocessor:
    """
    Legacy preprocessor class for pickle compatibility
    This class provides the methods needed by pickled preprocessor objects
    """
    
    def url_to_sequence(self, url):
        """Convert URL string to sequence of integers"""
        url = url.lower()
        return [self.char_to_idx.get(char, 0) for char in url]
    
    def encode_urls(self, urls):
        """
        Encode multiple URLs to padded sequences
        
        Args:
            urls: List of URL strings
            
        Returns:
            np.ndarray: Padded integer sequences
        """
        sequences = [self.url_to_sequence(url) for url in urls]
        padded = pad_sequences(
            sequences,
            maxlen=self.max_url_length,
            padding='post',
            truncating='post'
        )
        return padded
    
    def encode_domains(self, urls):
        """
        Extract and encode domains from URLs
        
        Args:
            urls: List of URL strings
            
        Returns:
            np.ndarray: Padded domain sequences
        """
        domains = [self.extract_domain(url) for url in urls]
        sequences = [self.url_to_sequence(domain) for domain in domains]
        padded = pad_sequences(
            sequences,
            maxlen=self.max_domain_length,
            padding='post',
            truncating='post'
        )
        return padded
    
    def extract_domain(self, url):
        """Extract domain from URL"""
        from urllib.parse import urlparse
        import re
        
        url = url.lower().strip()
        
        if not url.startswith(('http://', 'https://', '//')):
            url = 'http://' + url
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            if not domain:
                parts = parsed.path.split('/')
                if parts:
                    domain = parts[0]
            
            if ':' in domain:
                domain = domain.split(':')[0]
            
            if not domain:
                domain = url
            
            return domain
        except Exception:
            match = re.search(r'([a-z0-9\-\.]+\.[a-z]{2,}|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})', url)
            if match:
                return match.group(1)
            else:
                return url.split('/')[0]


def load_preprocessor(preprocessor_path: str):
    """
    Load preprocessor from pickle file
    
    Args:
        preprocessor_path: Path to preprocessor.pkl
        
    Returns:
        Preprocessor object with encode_urls(), encode_domains() methods
    """
    with open(preprocessor_path, 'rb') as f:
        return pickle.load(f)
