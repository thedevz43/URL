"""
International and Advanced Data Augmentation

Generates diverse malicious and benign URLs including:
1. International domains (.de, .co.uk, .jp, .fr, .ru, .cn)
2. IDN (Internationalized Domain Names) with Unicode
3. Advanced adversarial attacks
4. Regional TLD abuse patterns

Addresses model limitations:
- Weak performance on non-.com domains
- Missing international phishing patterns
- No IDN homograph training data
"""

import random
import numpy as np
import pandas as pd
from typing import List, Dict
import urllib.parse


class InternationalDataAugmentor:
    """Generate international and advanced adversarial URL samples"""
    
    # International top domains by region
    INTERNATIONAL_DOMAINS = {
        'de': [  # Germany
            'amazon.de', 'google.de', 'youtube.de', 'ebay.de', 'facebook.de',
            'gmx.net', 'web.de', 't-online.de', 'spiegel.de', 'bild.de',
            'heise.de', 'chip.de', 'wikipedia.de', 'reddit.de', 'zalando.de'
        ],
        'co.uk': [  # United Kingdom
            'bbc.co.uk', 'amazon.co.uk', 'ebay.co.uk', 'gov.uk', 'google.co.uk',
            'theguardian.com', 'dailymail.co.uk', 'telegraph.co.uk',
            'sky.com', 'argos.co.uk', 'tesco.com', 'bt.com', 'nhs.uk'
        ],
        'fr': [  # France
            'google.fr', 'amazon.fr', 'leboncoin.fr', 'orange.fr', 'free.fr',
            'lemonde.fr', 'youtube.fr', 'facebook.fr', 'wikipedia.fr',
            'lefigaro.fr', 'leparisien.fr', 'allocine.fr'
        ],
        'jp': [  # Japan
            'yahoo.co.jp', 'google.co.jp', 'amazon.co.jp', 'rakuten.co.jp',
            'youtube.co.jp', 'nicovideo.jp', 'line.me', 'pixiv.net',
            'dmm.com', 'fc2.com', 'ameblo.jp', 'livedoor.jp'
        ],
        'ru': [  # Russia
            'yandex.ru', 'vk.com', 'ok.ru', 'mail.ru', 'avito.ru',
            'aliexpress.ru', 'google.ru', 'youtube.ru', 'wikipedia.ru',
            'rambler.ru', 'lenta.ru', 'rbc.ru'
        ],
        'cn': [  # China
            'baidu.com', 'qq.com', 'taobao.com', 'tmall.com', 'jd.com',
            'sohu.com', 'sina.com.cn', 'weibo.com', 'zhihu.com',
            'bilibili.com', 'douban.com', '163.com'
        ],
        'es': [  # Spain
            'google.es', 'youtube.es', 'amazon.es', 'facebook.es',
            'elpais.com', 'marca.com', 'elmundo.es', 'elconfidencial.com'
        ],
        'it': [  # Italy
            'google.it', 'amazon.it', 'youtube.it', 'facebook.it',
            'repubblica.it', 'corriere.it', 'libero.it', 'virgilio.it'
        ],
        'au': [  # Australia
            'google.com.au', 'news.com.au', 'ebay.com.au', 'gumtree.com.au',
            'seek.com.au', 'realestate.com.au', 'bom.gov.au'
        ],
        'ca': [  # Canada
            'google.ca', 'amazon.ca', 'cbc.ca', 'globalnews.ca',
            'canada.ca', 'gc.ca', 'cra-arc.gc.ca'
        ],
        'br': [  # Brazil
            'google.com.br', 'globo.com', 'uol.com.br', 'mercadolivre.com.br',
            'youtube.com.br', 'facebook.com.br', 'terra.com.br'
        ],
        'in': [  # India
            'google.co.in', 'flipkart.com', 'amazon.in', 'paytm.com',
            'indiatimes.com', 'ndtv.com', 'timesofindia.com'
        ],
        'mx': [  # Mexico
            'google.com.mx', 'mercadolibre.com.mx', 'amazon.com.mx',
            'youtube.com.mx', 'facebook.com.mx'
        ]
    }
    
    # IDN homograph characters (lookalikes)
    IDN_HOMOGRAPHS = {
        'a': ['а', 'ạ', 'ą', 'ă', 'ǎ'],  # Cyrillic/Latin variations
        'e': ['е', 'ė', 'ę', 'ě', 'ẹ'],
        'o': ['о', 'ọ', 'ő', 'ŏ', 'ȯ'],
        'p': ['р', 'ṗ', 'ƿ'],
        'c': ['с', 'ċ', 'č', 'ç'],
        'x': ['х', 'ẋ', 'ӽ'],
        'y': ['у', 'ý', 'ÿ', 'ү'],
        'i': ['і', 'ı', 'ï', 'ì', 'í'],
        's': ['ѕ', 'ś', 'ş', 'š'],
        'h': ['һ', 'ḥ', 'ħ']
    }
    
    # Unicode scripts for IDN attacks
    UNICODE_SCRIPTS = {
        'cyrillic': 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
        'greek': 'αβγδεζηθικλμνξοπρστυφχψω',
        'arabic': 'ابتثجحخدذرزسشصضطظعغفقكلمنهوي',
        'hebrew': 'אבגדהוזחטיכלמנסעפצקרשת',
        'chinese': '的一是不了人我在有他这为之大来以个中上们',
        'japanese': 'あいうえおかきくけこさしすせそたちつてと'
    }
    
    # Regional malicious TLD patterns
    REGIONAL_ABUSE_PATTERNS = {
        'eastern_europe': ['.tk', '.ml', '.ga', '.cf', '.gq', '.ru', '.ua'],
        'asia': ['.tk', '.ml', '.cn', '.top', '.xyz', '.work'],
        'africa': ['.ml', '.ga', '.tk', '.cf', '.gq'],
        'generic_abuse': ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', 
                         '.work', '.click', '.link', '.download']
    }
    
    def __init__(self, seed=42):
        """Initialize augmentor"""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    # ========================================================================
    # BENIGN INTERNATIONAL URL GENERATION
    # ========================================================================
    
    def generate_benign_international(self, n_samples=10000) -> pd.DataFrame:
        """
        Generate legitimate URLs from international domains
        
        Args:
            n_samples: Number of URLs to generate
            
        Returns:
            DataFrame with columns: url, type
        """
        urls = []
        
        # Distribute across regions
        regions = list(self.INTERNATIONAL_DOMAINS.keys())
        samples_per_region = n_samples // len(regions)
        
        for region, domains in self.INTERNATIONAL_DOMAINS.items():
            for _ in range(samples_per_region):
                domain = random.choice(domains)
                
                # Generate diverse URL patterns
                patterns = [
                    f"https://{domain}",
                    f"https://{domain}/products",
                    f"https://{domain}/search?q=query",
                    f"https://www.{domain}/about",
                    f"https://{domain}/docs/reference",
                    f"https://support.{domain}/help",
                    f"https://{domain}/account/settings",
                    f"https://blog.{domain}/posts/{random.randint(1,1000)}",
                    f"https://{domain}/category/{random.choice(['tech', 'news', 'sports'])}",
                    f"https://api.{domain}/v{random.randint(1,3)}/users"
                ]
                
                url = random.choice(patterns)
                urls.append({'url': url, 'type': 'benign'})
        
        return pd.DataFrame(urls)
    
    # ========================================================================
    # IDN HOMOGRAPH ATTACK GENERATION
    # ========================================================================
    
    def generate_idn_homograph_attacks(self, n_samples=2000) -> pd.DataFrame:
        """
        Generate IDN homograph phishing attacks
        
        Creates URLs where legitimate domains are mimicked using
        visually similar Unicode characters from different scripts
        
        Args:
            n_samples: Number of attacks to generate
            
        Returns:
            DataFrame with attack URLs
        """
        urls = []
        
        # Target popular brands
        target_brands = [
            'paypal', 'google', 'amazon', 'facebook', 'apple',
            'microsoft', 'netflix', 'twitter', 'instagram', 'linkedin',
            'github', 'dropbox', 'spotify', 'adobe', 'salesforce'
        ]
        
        for _ in range(n_samples):
            brand = random.choice(target_brands)
            
            # Replace 1-3 characters with homographs
            num_replacements = random.randint(1, min(3, len(brand)))
            brand_chars = list(brand)
            
            for _ in range(num_replacements):
                # Find replaceable character
                replaceable_pos = [i for i, c in enumerate(brand_chars) 
                                  if c in self.IDN_HOMOGRAPHS]
                
                if replaceable_pos:
                    pos = random.choice(replaceable_pos)
                    original_char = brand_chars[pos]
                    homograph = random.choice(self.IDN_HOMOGRAPHS[original_char])
                    brand_chars[pos] = homograph
            
            fake_domain = ''.join(brand_chars)
            
            # Add suspicious path
            paths = [
                '/login.php',
                '/verify-account.html',
                '/security/update.asp',
                '/signin?redirect=account',
                '/confirm-identity.php',
                '/suspended-account/restore.html'
            ]
            
            url = f"https://{fake_domain}.com{random.choice(paths)}"
            urls.append({'url': url, 'type': 'phishing'})
        
        return pd.DataFrame(urls)
    
    # ========================================================================
    # INTERNATIONAL PHISHING GENERATION
    # ========================================================================
    
    def generate_international_phishing(self, n_samples=3000) -> pd.DataFrame:
        """
        Generate phishing URLs targeting international users
        
        Uses:
        - Regional TLDs
        - Local brand impersonation
        - Regional language patterns
        
        Args:
            n_samples: Number of phishing URLs to generate
            
        Returns:
            DataFrame with phishing URLs
        """
        urls = []
        
        # Regional phishing patterns
        regional_patterns = {
            'de': ['sicherheit', 'konto', 'bestätigung', 'aktualisierung'],
            'fr': ['securite', 'compte', 'confirmation', 'verification'],
            'es': ['seguridad', 'cuenta', 'confirmacion', 'verificar'],
            'ru': ['bezopasnost', 'account', 'proverka', 'obnovlenie'],
            'jp': ['security', 'account', 'verify', 'update'],
            'cn': ['security', 'account', 'verify', 'confirm']
        }
        
        for _ in range(n_samples):
            # Choose region
            region = random.choice(list(self.INTERNATIONAL_DOMAINS.keys()))
            legit_domain = random.choice(self.INTERNATIONAL_DOMAINS[region])
            
            # Extract base domain
            base = legit_domain.split('.')[0]
            
            # Create fake variation
            variations = [
                f"{base}-security",
                f"{base}-verify",
                f"{base}-account",
                f"{base}-support",
                f"secure-{base}",
                f"verify-{base}",
                f"{base}{random.randint(1,99)}"
            ]
            
            fake_base = random.choice(variations)
            
            # Use abused TLD
            abuse_tlds = self.REGIONAL_ABUSE_PATTERNS['generic_abuse']
            tld = random.choice(abuse_tlds)
            
            # Create URL
            paths = [
                '/login.php',
                '/signin.html',
                '/verify-account.asp',
                '/security-check.php',
                '/update-info.html'
            ]
            
            url = f"http://{fake_base}{tld}{random.choice(paths)}"
            urls.append({'url': url, 'type': 'phishing'})
        
        return pd.DataFrame(urls)
    
    # ========================================================================
    # ADVANCED ADVERSARIAL ATTACKS
    # ========================================================================
    
    def generate_advanced_adversarial(self, n_samples=2000) -> pd.DataFrame:
        """
        Generate sophisticated adversarial attacks
        
        Includes:
        - Unicode normalization attacks
        - Mixed-script attacks
        - Punycode obfuscation
        - Subdomain chaining with international domains
        
        Args:
            n_samples: Number of attacks to generate
            
        Returns:
            DataFrame with adversarial URls
        """
        urls = []
        
        attack_types = [
            self._generate_unicode_normalization_attack,
            self._generate_mixed_script_attack,
            self._generate_international_subdomain_chain,
            self._generate_unicode_path_attack
        ]
        
        samples_per_type = n_samples // len(attack_types)
        
        for attack_func in attack_types:
            for _ in range(samples_per_type):
                try:
                    url_data = attack_func()
                    urls.append(url_data)
                except:
                    pass  # Skip failed generations
        
        return pd.DataFrame(urls)
    
    def _generate_unicode_normalization_attack(self) -> Dict:
        """Unicode homograph with normalization tricks"""
        brand = random.choice(['paypal', 'amazon', 'google'])
        
        # Use combining characters
        modified = ''
        for char in brand:
            modified += char
            if random.random() < 0.3:
                # Add zero-width character or combining accent
                modified += random.choice(['\u200b', '\u200c', '\u0301', '\u0300'])
        
        url = f"https://{modified}.com/login"
        return {'url': url, 'type': 'phishing'}
    
    def _generate_mixed_script_attack(self) -> Dict:
        """Mix Latin with Cyrillic/Greek characters"""
        brand = random.choice(['microsoft', 'apple', 'facebook'])
        
        # Replace some characters with visually similar from different script
        chars = list(brand)
        for i, c in enumerate(chars):
            if c in self.IDN_HOMOGRAPHS and random.random() < 0.4:
                chars[i] = random.choice(self.IDN_HOMOGRAPHS[c])
        
        fake_domain = ''.join(chars)
        url = f"https://{fake_domain}.com/account/verify"
        return {'url': url, 'type': 'phishing'}
    
    def _generate_international_subdomain_chain(self) -> Dict:
        """Long subdomain chains hiding malicious intent"""
        region = random.choice(list(self.INTERNATIONAL_DOMAINS.keys()))
        legit = random.choice(self.INTERNATIONAL_DOMAINS[region])
        
        # Create long subdomain chain
        subdomains = []
        for _ in range(random.randint(3, 7)):
            subdomains.append(random.choice(['secure', 'verify', 'auth', 'login', 'account']))
        
        chain = '.'.join(subdomains)
        malicious_tld = random.choice(['.tk', '.ml', '.ga'])
        
        url = f"https://{chain}.{legit.replace('.', '-')}{malicious_tld}/confirm.php"
        return {'url': url, 'type': 'phishing'}
    
    def _generate_unicode_path_attack(self) -> Dict:
        """Unicode characters in path to evade detection"""
        scripts = ['cyrillic', 'greek', 'arabic']
        script = random.choice(scripts)
        unicode_chars = random.sample(self.UNICODE_SCRIPTS[script], k=5)
        unicode_path = ''.join(unicode_chars)
        
        domain = random.choice(['example', 'test', 'site']) + random.choice(['.tk', '.ml'])
        url = f"http://{domain}/{unicode_path}/malware.exe"
        return {'url': url, 'type': 'malware'}
    
    # ========================================================================
    # COMPLETE AUGMENTATION PIPELINE
    # ========================================================================
    
    def generate_complete_international_dataset(
        self,
        n_benign_intl=10000,
        n_idn_attacks=2000,
        n_intl_phishing=3000,
        n_adversarial=2000
    ) -> pd.DataFrame:
        """
        Generate complete international augmentation dataset
        
        Args:
            n_benign_intl: Benign international URLs
            n_idn_attacks: IDN homograph attacks
            n_intl_phishing: International phishing
            n_adversarial: Advanced adversarial attacks
            
        Returns:
            Combined DataFrame
        """
        print("="*80)
        print("GENERATING INTERNATIONAL AUGMENTATION DATASET")
        print("="*80)
        print()
        
        datasets = []
        
        # 1. Benign international
        print(f"[1/4] Generating {n_benign_intl:,} benign international URLs...")
        df_benign = self.generate_benign_international(n_benign_intl)
        datasets.append(df_benign)
        print(f"      ✓ Generated {len(df_benign):,} URLs")
        
        # 2. IDN homograph attacks
        print(f"[2/4] Generating {n_idn_attacks:,} IDN homograph attacks...")
        df_idn = self.generate_idn_homograph_attacks(n_idn_attacks)
        datasets.append(df_idn)
        print(f"      ✓ Generated {len(df_idn):,} attacks")
        
        # 3. International phishing
        print(f"[3/4] Generating {n_intl_phishing:,} international phishing URLs...")
        df_phishing = self.generate_international_phishing(n_intl_phishing)
        datasets.append(df_phishing)
        print(f"      ✓ Generated {len(df_phishing):,} phishing URLs")
        
        # 4. Advanced adversarial
        print(f"[4/4] Generating {n_adversarial:,} advanced adversarial attacks...")
        df_adversarial = self.generate_advanced_adversarial(n_adversarial)
        datasets.append(df_adversarial)
        print(f"      ✓ Generated {len(df_adversarial):,} adversarial URLs")
        
        # Combine all
        print()
        print("Combining datasets...")
        df_combined = pd.concat(datasets, ignore_index=True)
        
        # Remove duplicates
        initial_count = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['url'])
        final_count = len(df_combined)
        
        if initial_count > final_count:
            print(f"Removed {initial_count - final_count:,} duplicates")
        
        print()
        print("="*80)
        print("INTERNATIONAL AUGMENTATION COMPLETE")
        print("="*80)
        print(f"Total URLs generated: {final_count:,}")
        print()
        print("Distribution:")
        for label, count in df_combined['type'].value_counts().items():
            pct = 100 * count / final_count
            print(f"  {label:15} : {count:6,} ({pct:5.2f}%)")
        print()
        
        return df_combined


if __name__ == "__main__":
    """Test international augmentation"""
    
    augmentor = InternationalDataAugmentor(seed=42)
    
    # Generate complete dataset
    df_intl = augmentor.generate_complete_international_dataset(
        n_benign_intl=1000,
        n_idn_attacks=200,
        n_intl_phishing=300,
        n_adversarial=200
    )
    
    print("Sample URLs:")
    print()
    for _, row in df_intl.sample(10).iterrows():
        print(f"{row['type']:12} : {row['url'][:70]}")
    
    # Save
    output_path = 'data/international_augmentation.csv'
    df_intl.to_csv(output_path, index=False)
    print()
    print(f"✓ Saved to {output_path}")