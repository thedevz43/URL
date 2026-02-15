"""
Domain Reputation Scoring System

Simulates Tranco Top 1000 + additional trusted domains
Provides reputation scores for inference-time adjustment
"""

import re
from urllib.parse import urlparse
from typing import Dict, Optional
import numpy as np


class DomainReputationScorer:
    """
    Domain reputation scoring for false positive mitigation
    """
    
    def __init__(self):
        self.reputation_db = self._load_reputation_database()
        self.tld_risk_scores = self._load_tld_risk_scores()
        
    def _load_reputation_database(self) -> Dict[str, float]:
        """
        Simulate Tranco Top 1000 + additional trusted domains
        Reputation score: 0.0 (unknown) to 1.0 (highest reputation)
        """
        # Top 100 most trusted domains (simulating Tranco Top 100)
        top_100 = [
            'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
            'linkedin.com', 'wikipedia.org', 'reddit.com', 'amazon.com', 'ebay.com',
            'microsoft.com', 'apple.com', 'netflix.com', 'zoom.us', 'paypal.com',
            'github.com', 'stackoverflow.com', 'twitch.tv', 'nytimes.com', 'cnn.com',
            'bbc.com', 'bbc.co.uk', 'wordpress.com', 'adobe.com', 'salesforce.com',
            'oracle.com', 'ibm.com', 'intel.com', 'samsung.com', 'sony.com',
            'walmart.com', 'target.com', 'costco.com', 'homedepot.com', 'bestbuy.com',
            'spotify.com', 'dropbox.com', 'slack.com', 'zoom.com', 'discord.com',
            'whatsapp.com', 'telegram.org', 'signal.org', 'gmail.com', 'outlook.com',
            'yahoo.com', 'bing.com', 'duckduckgo.com', 'baidu.com', 'yandex.com',
            'chase.com', 'bankofamerica.com', 'wellsfargo.com', 'citi.com', 'capitalone.com',
            'usbank.com', 'pnc.com', 'discover.com', 'americanexpress.com', 'visa.com',
            'mastercard.com', 'stripe.com', 'square.com', 'venmo.com', 'cashapp.com',
            'coursera.org', 'udemy.com', 'edx.org', 'khanacademy.org', 'duolingo.com',
            'stanford.edu', 'mit.edu', 'harvard.edu', 'berkeley.edu', 'oxford.ac.uk',
            'espn.com', 'nba.com', 'nfl.com', 'mlb.com', 'fifa.com',
            'airbnb.com', 'booking.com', 'expedia.com', 'tripadvisor.com', 'hotels.com',
            'nvidia.com', 'amd.com', 'unity.com', 'unrealengine.com', 'steam.com',
            'playstation.com', 'xbox.com', 'nintendo.com', 'roblox.com', 'minecraft.net',
            'pinterest.com', 'tumblr.com', 'medium.com', 'quora.com', 'flickr.com'
        ]
        
        # Top 101-500 (high reputation)
        high_reputation = [
            'cloudflare.com', 'vimeo.com', 'soundcloud.com', 'etsy.com', 'shopify.com',
            'wix.com', 'squarespace.com', 'godaddy.com', 'namecheap.com', 'hostgator.com',
            'bluehost.com', 'digitalocean.com', 'linode.com', 'heroku.com', 'netlify.com',
            'vercel.com', 'aws.amazon.com', 'azure.microsoft.com', 'cloud.google.com',
            'docker.com', 'kubernetes.io', 'jenkins.io', 'gitlab.com', 'bitbucket.org',
            'atlassian.com', 'jira.com', 'confluence.com', 'trello.com', 'asana.com',
            'monday.com', 'notion.so', 'evernote.com', 'onenote.com', 'googledrive.com',
        ]
        
        # Top 501-1000 (moderate reputation)
        moderate_reputation = [
            'fedex.com', 'ups.com', 'usps.com', 'dhl.com', 'ikea.com',
            'lowes.com', 'macys.com', 'nordstrom.com', 'gap.com', 'hm.com',
            'zara.com', 'uniqlo.com', 'nike.com', 'adidas.com', 'puma.com',
            'reebok.com', 'underarmour.com', 'patagonia.com', 'thenorthface.com',
        ]
        
        reputation_db = {}
        
        # Assign scores (0.9-1.0 for top 100, 0.7-0.89 for 101-500, 0.5-0.69 for 501-1000)
        for i, domain in enumerate(top_100):
            reputation_db[domain] = 0.95 + (100 - i) / 100 * 0.05  # 0.95-1.0
            
        for i, domain in enumerate(high_reputation):
            reputation_db[domain] = 0.75 + (len(high_reputation) - i) / len(high_reputation) * 0.14  # 0.75-0.89
            
        for i, domain in enumerate(moderate_reputation):
            reputation_db[domain] = 0.55 + (len(moderate_reputation) - i) / len(moderate_reputation) * 0.14  # 0.55-0.69
        
        return reputation_db
    
    def _load_tld_risk_scores(self) -> Dict[str, float]:
        """
        TLD risk scoring
        Risk score: 0.0 (safe) to 1.0 (high risk)
        """
        return {
            # Trusted TLDs (low risk)
            'com': 0.1, 'org': 0.1, 'net': 0.15, 'edu': 0.05, 'gov': 0.05,
            'mil': 0.05, 'int': 0.1,
            
            # Country TLDs (low-medium risk)
            'uk': 0.1, 'de': 0.1, 'fr': 0.1, 'ca': 0.1, 'au': 0.1,
            'jp': 0.1, 'cn': 0.2, 'ru': 0.3, 'br': 0.15, 'in': 0.15,
            
            # New gTLDs (medium-high risk)
            'xyz': 0.5, 'top': 0.6, 'loan': 0.8, 'work': 0.5, 'click': 0.7,
            'gq': 0.9, 'ml': 0.9, 'ga': 0.9, 'cf': 0.9, 'tk': 0.9,
            
            # Tech TLDs (low-medium risk)
            'io': 0.2, 'dev': 0.15, 'ai': 0.2, 'tech': 0.3, 'app': 0.2,
        }
    
    def extract_domain(self, url: str) -> Optional[str]:
        """
        Extract base domain from URL
        """
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Remove port
            if ':' in domain:
                domain = domain.split(':')[0]
            
            return domain if domain else None
            
        except Exception:
            return None
    
    def get_reputation_score(self, url: str) -> float:
        """
        Get reputation score for a URL
        
        Returns:
            float: Reputation score (0.0-1.0), where 1.0 = highest reputation
        """
        domain = self.extract_domain(url)
        if not domain:
            return 0.0
        
        # Check exact match
        if domain in self.reputation_db:
            return self.reputation_db[domain]
        
        # Check parent domain (e.g., mail.google.com -> google.com)
        parts = domain.split('.')
        if len(parts) > 2:
            parent_domain = '.'.join(parts[-2:])
            if parent_domain in self.reputation_db:
                # Subdomain gets slightly lower score than parent
                return self.reputation_db[parent_domain] * 0.95
        
        # Unknown domain
        return 0.0
    
    def get_tld_risk(self, url: str) -> float:
        """
        Get TLD risk score for a URL
        
        Returns:
            float: Risk score (0.0-1.0), where 1.0 = highest risk
        """
        domain = self.extract_domain(url)
        if not domain:
            return 0.5  # Unknown = medium risk
        
        # Extract TLD
        parts = domain.split('.')
        if len(parts) < 2:
            return 0.5
        
        tld = parts[-1]
        
        return self.tld_risk_scores.get(tld, 0.4)  # Default medium risk
    
    def is_trusted_domain(self, url: str, threshold: float = 0.7) -> bool:
        """
        Check if domain is trusted (reputation above threshold)
        """
        return self.get_reputation_score(url) >= threshold
    
    def get_domain_features(self, url: str) -> Dict[str, float]:
        """
        Get all domain-based features for adjustment
        """
        return {
            'reputation': self.get_reputation_score(url),
            'tld_risk': self.get_tld_risk(url),
            'is_trusted': self.is_trusted_domain(url),
        }


# Global instance
_reputation_scorer = None

def get_reputation_scorer() -> DomainReputationScorer:
    """Get global reputation scorer instance"""
    global _reputation_scorer
    if _reputation_scorer is None:
        _reputation_scorer = DomainReputationScorer()
    return _reputation_scorer
