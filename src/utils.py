"""
Utility functions for production inference
"""

import re
from urllib.parse import urlparse
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DomainReputationScorer:
    """Domain reputation scoring for FP mitigation"""
    
    def __init__(self):
        self.reputation_db = self._load_reputation_database()
    
    def _load_reputation_database(self) -> Dict[str, float]:
        """
        Tranco Top 1000 simulation
        Score: 0.0 (unknown) to 1.0 (highest reputation)
        """
        top_domains = [
            'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
            'linkedin.com', 'wikipedia.org', 'reddit.com', 'amazon.com', 'ebay.com',
            'microsoft.com', 'apple.com', 'netflix.com', 'zoom.us', 'paypal.com',
            'github.com', 'stackoverflow.com', 'twitch.tv', 'nytimes.com', 'cnn.com',
            'bbc.com', 'bbc.co.uk', 'wordpress.com', 'adobe.com', 'salesforce.com',
            'oracle.com', 'ibm.com', 'intel.com', 'samsung.com', 'sony.com',
            'walmart.com', 'target.com', 'costco.com', 'homedepot.com', 'bestbuy.com',
            'spotify.com', 'dropbox.com', 'slack.com', 'discord.com',
            'whatsapp.com', 'telegram.org', 'signal.org', 'gmail.com', 'outlook.com',
            'yahoo.com', 'bing.com', 'duckduckgo.com', 'baidu.com', 'yandex.com',
            'chase.com', 'bankofamerica.com', 'wellsfargo.com', 'citi.com', 'capitalone.com',
            'usbank.com', 'pnc.com', 'discover.com', 'americanexpress.com', 'visa.com',
            'mastercard.com', 'stripe.com', 'square.com', 'venmo.com',
            'coursera.org', 'udemy.com', 'edx.org', 'khanacademy.org', 'duolingo.com',
            'stanford.edu', 'mit.edu', 'harvard.edu', 'berkeley.edu', 'oxford.ac.uk',
            'espn.com', 'nba.com', 'nfl.com', 'mlb.com', 'fifa.com',
            'airbnb.com', 'booking.com', 'expedia.com', 'tripadvisor.com', 'hotels.com',
            'nvidia.com', 'amd.com', 'unity.com', 'steam.com',
            'playstation.com', 'xbox.com', 'nintendo.com', 'roblox.com',
            'pinterest.com', 'tumblr.com', 'medium.com', 'quora.com'
        ]
        
        reputation_db = {}
        
        # Top 50: reputation 1.0 (highest trust)
        for domain in top_domains[:50]:
            reputation_db[domain] = 1.0
        
        # Next 50: reputation 0.95 (elite)
        for domain in top_domains[50:100]:
            reputation_db[domain] = 0.95
        
        # Remaining: reputation 0.90-0.95 (trusted)
        for i, domain in enumerate(top_domains[100:]):
            reputation_db[domain] = 0.95 - (i * 0.001)
        
        return reputation_db
    
    def get_reputation_score(self, url: str) -> float:
        """
        Get reputation score for URL
        
        Args:
            url: URL to score
            
        Returns:
            Reputation score (0.0 to 1.0)
        """
        domain = self.extract_domain(url)
        if not domain:
            return 0.0
        
        return self.reputation_db.get(domain, 0.0)
    
    def extract_domain(self, url: str) -> Optional[str]:
        """
        Extract domain from URL
        
        Args:
            url: Input URL
            
        Returns:
            Domain string or None
        """
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]
            
            return domain if domain else None
            
        except Exception as e:
            logger.warning(f"Failed to extract domain from {url}: {str(e)}")
            return None


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for production
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    logging_config = {
        'level': getattr(logging, log_level.upper()),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        logging_config['filename'] = log_file
        logging_config['filemode'] = 'a'
    
    logging.basicConfig(**logging_config)
