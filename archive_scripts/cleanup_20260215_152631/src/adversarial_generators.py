"""
Adversarial URL Generators for Stress Testing
Generates malicious-looking URLs to test model robustness
"""

import random
import string
from typing import List, Dict
import unicodedata


class AdversarialURLGenerator:
    """Generate adversarial URLs for testing model robustness"""
    
    # Homoglyph mappings (visually similar characters)
    HOMOGLYPHS = {
        'a': ['а', 'ạ', 'ą', 'α', 'ä', 'à'],  # Cyrillic а, various accented
        'e': ['е', 'ę', 'ė', 'ë', 'è', 'é'],
        'o': ['о', 'ο', 'ö', 'ò', 'ó', 'ø'],
        'i': ['і', 'ı', 'ï', 'í', 'ì'],
        'c': ['с', 'ç', 'ć'],
        'p': ['р', 'þ'],
        'm': ['м', 'ṃ'],
        'h': ['һ', 'ḥ'],
        's': ['ѕ', 'ś', 'š'],
        'x': ['х', 'χ'],
        'y': ['у', 'ý', 'ÿ'],
        'n': ['п', 'ñ', 'ń'],
        'u': ['υ', 'ü', 'ù', 'ú'],
        'g': ['ģ', 'ğ'],
        'l': ['ӏ', 'ł'],
        'r': ['г', 'ř'],
        'k': ['κ', 'ķ'],
        't': ['τ', 'ţ']
    }
    
    # Major brands to impersonate
    BRANDS = [
        'google', 'amazon', 'microsoft', 'apple', 'facebook',
        'netflix', 'paypal', 'github', 'linkedin', 'twitter',
        'instagram', 'dropbox', 'adobe', 'yahoo', 'ebay',
        'bankofamerica', 'chase', 'wellsfargo', 'citibank',
        'americanexpress', 'visa', 'mastercard'
    ]
    
    # Common phishing paths
    PHISHING_PATHS = [
        '/login', '/signin', '/account/verify', '/secure/update',
        '/user/confirm', '/reset-password', '/billing/update',
        '/security/verify', '/account/suspended', '/verify-identity',
        '/claim-prize', '/gift-card', '/free-offer', '/limited-time',
        '/urgent-action', '/account-locked', '/payment-failed'
    ]
    
    # Suspicious TLDs
    SUSPICIOUS_TLDS = [
        '.tk', '.ml', '.ga', '.cf', '.gq', '.pw', '.cc', '.biz',
        '.info', '.xyz', '.top', '.win', '.download', '.link',
        '.click', '.loan', '.racing', '.accountant'
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed"""
        random.seed(seed)
    
    def generate_homoglyph_attack(self, brand: str = None) -> Dict[str, str]:
        """
        Generate URL with homoglyph character substitution
        
        Example: 'google.com' -> 'gооgle.com' (Cyrillic о)
        """
        if brand is None:
            brand = random.choice(self.BRANDS)
        
        # Substitute 1-3 characters with homoglyphs
        brand_list = list(brand)
        num_substitutions = random.randint(1, min(3, len(brand)))
        positions = random.sample(range(len(brand)), num_substitutions)
        
        for pos in positions:
            char = brand_list[pos].lower()
            if char in self.HOMOGLYPHS:
                brand_list[pos] = random.choice(self.HOMOGLYPHS[char])
        
        fake_brand = ''.join(brand_list)
        path = random.choice(self.PHISHING_PATHS)
        
        return {
            'url': f"https://{fake_brand}.com{path}",
            'type': 'homoglyph',
            'target_brand': brand,
            'substitutions': num_substitutions
        }
    
    def generate_subdomain_chain(self, brand: str = None) -> Dict[str, str]:
        """
        Generate URL with long subdomain chain to hide malicious domain
        
        Example: 'user.account.verify.google.com.evil-phishing.tk'
        """
        if brand is None:
            brand = random.choice(self.BRANDS)
        
        # Create legitimate-looking subdomain chain
        legit_subdomains = ['login', 'secure', 'account', 'verify', 'auth', 'user']
        num_subdomains = random.randint(2, 5)
        chain = random.sample(legit_subdomains, min(num_subdomains, len(legit_subdomains)))
        
        # Add brand as fake subdomain
        chain.append(brand)
        chain.append('com')
        
        # Real malicious domain
        evil_domain = self._generate_random_string(8, 12)
        evil_tld = random.choice(self.SUSPICIOUS_TLDS)
        
        full_domain = '.'.join(chain) + '.' + evil_domain + evil_tld
        path = random.choice(self.PHISHING_PATHS)
        
        return {
            'url': f"https://{full_domain}{path}",
            'type': 'subdomain_chain',
            'target_brand': brand,
            'chain_length': len(chain) + 1
        }
    
    def generate_typosquatting(self, brand: str = None) -> Dict[str, str]:
        """
        Generate URL with common typos
        
        Types: character omission, duplication, adjacent key, swap
        """
        if brand is None:
            brand = random.choice(self.BRANDS)
        
        typo_type = random.choice(['omit', 'duplicate', 'swap', 'add'])
        brand_list = list(brand)
        
        if typo_type == 'omit' and len(brand) > 4:
            # Remove a character
            pos = random.randint(1, len(brand) - 2)
            brand_list.pop(pos)
            typo_desc = 'character omission'
        
        elif typo_type == 'duplicate':
            # Duplicate a character
            pos = random.randint(0, len(brand) - 1)
            brand_list.insert(pos, brand_list[pos])
            typo_desc = 'character duplication'
        
        elif typo_type == 'swap' and len(brand) > 2:
            # Swap adjacent characters
            pos = random.randint(0, len(brand) - 2)
            brand_list[pos], brand_list[pos + 1] = brand_list[pos + 1], brand_list[pos]
            typo_desc = 'character swap'
        
        else:  # add
            # Add random character
            pos = random.randint(0, len(brand))
            brand_list.insert(pos, random.choice(string.ascii_lowercase))
            typo_desc = 'character addition'
        
        fake_brand = ''.join(brand_list)
        path = random.choice(self.PHISHING_PATHS)
        tld = random.choice(['.com', '.net'] + self.SUSPICIOUS_TLDS[:3])
        
        return {
            'url': f"https://{fake_brand}{tld}{path}",
            'type': 'typosquatting',
            'typo_method': typo_desc,
            'target_brand': brand
        }
    
    def generate_combosquatting(self, brand: str = None) -> Dict[str, str]:
        """
        Generate URL with brand + keyword combination
        
        Example: 'paypal-secure.com', 'login-amazon.com'
        """
        if brand is None:
            brand = random.choice(self.BRANDS)
        
        keywords = [
            'login', 'secure', 'verify', 'account', 'official',
            'support', 'help', 'service', 'online', 'signin',
            'auth', 'security', 'update', 'confirm'
        ]
        
        keyword = random.choice(keywords)
        separator = random.choice(['-', '', '.'])
        
        # Various patterns
        patterns = [
            f"{keyword}{separator}{brand}",
            f"{brand}{separator}{keyword}",
            f"{brand}{keyword}",
            f"{keyword}{brand}"
        ]
        
        fake_domain = random.choice(patterns)
        tld = random.choice(['.com', '.net'] + self.SUSPICIOUS_TLDS[:5])
        path = random.choice(self.PHISHING_PATHS)
        
        return {
            'url': f"https://{fake_domain}{tld}{path}",
            'type': 'combosquatting',
            'target_brand': brand,
            'keyword': keyword
        }
    
    def generate_idn_homograph(self, brand: str = None) -> Dict[str, str]:
        """
        Generate Internationalized Domain Name (IDN) homograph attack
        Mix scripts (Latin + Cyrillic/Greek)
        """
        if brand is None:
            brand = random.choice(self.BRANDS)
        
        # Replace multiple characters with unicode lookalikes
        brand_list = list(brand)
        for i, char in enumerate(brand_list):
            if char.lower() in self.HOMOGLYPHS and random.random() < 0.4:
                brand_list[i] = random.choice(self.HOMOGLYPHS[char.lower()])
        
        fake_brand = ''.join(brand_list)
        path = random.choice(self.PHISHING_PATHS)
        
        return {
            'url': f"https://{fake_brand}.com{path}",
            'type': 'idn_homograph',
            'target_brand': brand,
            'unicode': True
        }
    
    def generate_fake_subdomain(self, brand: str = None) -> Dict[str, str]:
        """
        Generate URL with brand as subdomain of fake domain
        
        Example: 'amazon.login-verify.tk'
        """
        if brand is None:
            brand = random.choice(self.BRANDS)
        
        fake_base = random.choice([
            'login-verify', 'account-secure', 'signin-auth',
            'verify-account', 'secure-login', 'auth-service'
        ])
        
        tld = random.choice(self.SUSPICIOUS_TLDS)
        path = random.choice(self.PHISHING_PATHS)
        
        return {
            'url': f"https://{brand}.{fake_base}{tld}{path}",
            'type': 'fake_subdomain',
            'target_brand': brand
        }
    
    def generate_url_shortener_disguise(self) -> Dict[str, str]:
        """
        Generate fake URL shortener that looks legitimate
        """
        shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co']
        base = random.choice(shorteners).split('.')[0]
        
        # Slightly modified shortener
        modifications = [
            f"{base}1.ly",
            f"{base}-link.com",
            f"{base}url.com",
            f"{base}.co"
        ]
        
        fake_shortener = random.choice(modifications)
        short_code = self._generate_random_string(6, 8, include_digits=True)
        
        return {
            'url': f"https://{fake_shortener}/{short_code}",
            'type': 'fake_shortener',
            'impersonates': base
        }
    
    def generate_data_uri_attack(self) -> Dict[str, str]:
        """Generate data URI scheme attack"""
        return {
            'url': 'data:text/html,<script>alert("xss")</script>',
            'type': 'data_uri',
            'target_brand': 'N/A'
        }
    
    def generate_ip_obfuscation(self) -> Dict[str, str]:
        """
        Generate obfuscated IP addresses
        
        Examples: decimal, octal, hex representations
        """
        # Random IP
        ip_parts = [random.randint(1, 254) for _ in range(4)]
        
        obfuscation = random.choice(['decimal', 'octal', 'hex', 'mixed'])
        
        if obfuscation == 'decimal':
            # Convert to single decimal number
            decimal = (ip_parts[0] << 24) + (ip_parts[1] << 16) + \
                     (ip_parts[2] << 8) + ip_parts[3]
            url = f"http://{decimal}/login"
        
        elif obfuscation == 'octal':
            # Octal representation
            octal_parts = [f"0{oct(part)[2:]}" for part in ip_parts]
            url = f"http://{'.'.join(octal_parts)}/login"
        
        elif obfuscation == 'hex':
            # Hex representation
            hex_parts = [f"0x{hex(part)[2:]}" for part in ip_parts]
            url = f"http://{'.'.join(hex_parts)}/login"
        
        else:  # mixed
            # Mix decimal and octal
            mixed = [
                str(ip_parts[0]),
                f"0{oct(ip_parts[1])[2:]}",
                str(ip_parts[2]),
                f"0x{hex(ip_parts[3])[2:]}"
            ]
            url = f"http://{'.'.join(mixed)}/login"
        
        return {
            'url': url,
            'type': 'ip_obfuscation',
            'method': obfuscation
        }
    
    def _generate_random_string(self, min_len: int, max_len: int, 
                                include_digits: bool = False) -> str:
        """Generate random string for domain names"""
        length = random.randint(min_len, max_len)
        chars = string.ascii_lowercase
        if include_digits:
            chars += string.digits
        return ''.join(random.choice(chars) for _ in range(length))
    
    def generate_adversarial_batch(self, n_samples: int = 100) -> List[Dict[str, str]]:
        """
        Generate batch of adversarial URLs covering all attack types
        
        Returns list of dicts with 'url', 'type', and metadata
        """
        generators = [
            self.generate_homoglyph_attack,
            self.generate_subdomain_chain,
            self.generate_typosquatting,
            self.generate_combosquatting,
            self.generate_idn_homograph,
            self.generate_fake_subdomain,
            self.generate_url_shortener_disguise,
            self.generate_ip_obfuscation,
        ]
        
        adversarial_urls = []
        
        # Distribute samples across different attack types
        samples_per_type = n_samples // len(generators)
        remainder = n_samples % len(generators)
        
        for i, generator in enumerate(generators):
            count = samples_per_type + (1 if i < remainder else 0)
            for _ in range(count):
                try:
                    url_data = generator()
                    adversarial_urls.append(url_data)
                except Exception as e:
                    print(f"Warning: Failed to generate with {generator.__name__}: {e}")
        
        return adversarial_urls


class BrandDomainGenerator:
    """Generate legitimate brand domains for false positive testing"""
    
    # Top 100 domains (simulate Alexa/Tranco rankings)
    TOP_DOMAINS = [
        'google.com', 'youtube.com', 'facebook.com', 'amazon.com',
        'wikipedia.org', 'twitter.com', 'instagram.com', 'linkedin.com',
        'reddit.com', 'netflix.com', 'microsoft.com', 'apple.com',
        'github.com', 'stackoverflow.com', 'docs.python.org',
        'npmjs.com', 'medium.com', 'quora.com', 'pinterest.com',
        'tumblr.com', 'wordpress.com', 'blogger.com', 'vimeo.com',
        'flickr.com', 'soundcloud.com', 'spotify.com', 'twitch.tv',
        'dropbox.com', 'adobe.com', 'salesforce.com', 'oracle.com',
        'ibm.com', 'intel.com', 'nvidia.com', 'cisco.com',
        'paypal.com', 'ebay.com', 'aliexpress.com', 'walmart.com',
        'target.com', 'bestbuy.com', 'homedepot.com', 'ikea.com',
        'bankofamerica.com', 'chase.com', 'wellsfargo.com',
        'citibank.com', 'capitalone.com', 'discover.com',
        'nytimes.com', 'washingtonpost.com', 'bbc.com', 'cnn.com',
        'forbes.com', 'bloomberg.com', 'reuters.com', 'theguardian.com',
        'weather.com', 'espn.com', 'nfl.com', 'nba.com',
        'zoom.us', 'slack.com', 'trello.com', 'asana.com',
        'notion.so', 'discord.com', 'telegram.org', 'whatsapp.com',
        'office.com', 'outlook.com', 'gmail.com', 'protonmail.com',
        'unity.com', 'unrealengine.com', 'steam.com', 'epicgames.com',
        'mozilla.org', 'firefox.com', 'chrome.com', 'edge.com',
        'stackoverflow.com', 'stackexchange.com', 'kaggle.com',
        'coursera.org', 'udemy.com', 'edx.org', 'khanacademy.org',
        'mit.edu', 'stanford.edu', 'harvard.edu', 'berkeley.edu',
        'arxiv.org', 'researchgate.net', 'scholar.google.com',
        'docker.com', 'kubernetes.io', 'terraform.io', 'ansible.com'
    ]
    
    COMMON_PATHS = [
        '/', '/about', '/contact', '/help', '/support', '/docs',
        '/blog', '/news', '/products', '/pricing', '/login', '/signup',
        '/account', '/settings', '/profile', '/dashboard', '/search',
        '/user/profile', '/article/12345', '/post/abc123',
        '/category/technology', '/tag/python', '/download',
        '/api/v1/users', '/static/css/style.css', '/images/logo.png'
    ]
    
    def generate_legitimate_urls(self, n_samples: int = 50) -> List[Dict[str, str]]:
        """Generate legitimate URLs from top domains"""
        urls = []
        
        for _ in range(n_samples):
            domain = random.choice(self.TOP_DOMAINS)
            path = random.choice(self.COMMON_PATHS)
            protocol = random.choice(['https', 'http'])
            
            # Add query parameters sometimes
            if random.random() < 0.3:
                params = f"?id={random.randint(1, 999999)}"
                path += params
            
            urls.append({
                'url': f"{protocol}://{domain}{path}",
                'domain': domain,
                'expected_class': 'benign'
            })
        
        return urls


if __name__ == '__main__':
    # Demo usage
    print("="*80)
    print("ADVERSARIAL URL GENERATOR DEMO")
    print("="*80)
    
    gen = AdversarialURLGenerator(seed=42)
    
    print("\n1. Homoglyph Attack:")
    print(gen.generate_homoglyph_attack('github'))
    
    print("\n2. Subdomain Chain:")
    print(gen.generate_subdomain_chain('paypal'))
    
    print("\n3. Typosquatting:")
    print(gen.generate_typosquatting('amazon'))
    
    print("\n4. Combosquatting:")
    print(gen.generate_combosquatting('microsoft'))
    
    print("\n5. IP Obfuscation:")
    print(gen.generate_ip_obfuscation())
    
    print(f"\n\nGenerating batch of 20 adversarial URLs...")
    batch = gen.generate_adversarial_batch(20)
    for i, item in enumerate(batch[:5], 1):
        print(f"\n{i}. Type: {item['type']}")
        print(f"   URL: {item['url']}")
    
    print("\n" + "="*80)
    print("BRAND DOMAIN GENERATOR DEMO")
    print("="*80)
    
    brand_gen = BrandDomainGenerator()
    legit_urls = brand_gen.generate_legitimate_urls(5)
    for i, item in enumerate(legit_urls, 1):
        print(f"\n{i}. Domain: {item['domain']}")
        print(f"   URL: {item['url']}")
