"""
Data Augmentation Script for Brand Domain URLs

Generates realistic benign URLs from top-ranked domains to address
the 82% false positive rate on legitimate brand sites.

Strategy:
- Simulate Tranco top 1000 domains
- Generate diverse URL patterns (login, docs, products, queries)
- Create 20,000+ high-quality benign samples
"""

import random
import pandas as pd
import urllib.parse
from typing import List, Dict
import numpy as np


class BrandURLAugmentor:
    """Generate realistic benign URLs from top domains"""
    
    # Simulated Tranco top 1000 domains (representative sample)
    TOP_DOMAINS = [
        # Tech Giants
        'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
        'linkedin.com', 'reddit.com', 'wikipedia.org', 'amazon.com', 'apple.com',
        'microsoft.com', 'netflix.com', 'zoom.us', 'twitch.tv', 'tiktok.com',
        
        # E-commerce
        'ebay.com', 'aliexpress.com', 'walmart.com', 'target.com', 'etsy.com',
        'shopify.com', 'bestbuy.com', 'wayfair.com', 'overstock.com', 'newegg.com',
        
        # Cloud & Developer
        'github.com', 'gitlab.com', 'stackoverflow.com', 'bitbucket.org', 'npmjs.com',
        'pypi.org', 'docker.com', 'kubernetes.io', 'cloudflare.com', 'aws.amazon.com',
        'azure.microsoft.com', 'cloud.google.com', 'heroku.com', 'digitalocean.com',
        'vercel.com', 'netlify.com', 'firebase.google.com', 'mongodb.com',
        
        # Business & Productivity
        'salesforce.com', 'slack.com', 'atlassian.com', 'notion.so', 'trello.com',
        'asana.com', 'monday.com', 'clickup.com', 'airtable.com', 'dropbox.com',
        'box.com', 'onedrive.live.com', 'drive.google.com', 'evernote.com',
        
        # Media & Entertainment
        'spotify.com', 'soundcloud.com', 'vimeo.com', 'dailymotion.com', 'hulu.com',
        'disneyplus.com', 'hbomax.com', 'crunchyroll.com', 'twitch.tv', 'imgur.com',
        'flickr.com', 'deviantart.com', 'artstation.com', 'behance.net',
        
        # News & Information
        'cnn.com', 'bbc.com', 'nytimes.com', 'theguardian.com', 'washingtonpost.com',
        'forbes.com', 'bloomberg.com', 'reuters.com', 'techcrunch.com', 'wired.com',
        'medium.com', 'substack.com', 'wordpress.com', 'blogger.com',
        
        # Education
        'coursera.org', 'udemy.com', 'edx.org', 'khanacademy.org', 'udacity.com',
        'skillshare.com', 'duolingo.com', 'codecademy.com', 'pluralsight.com',
        'mit.edu', 'stanford.edu', 'harvard.edu', 'berkeley.edu', 'oxford.ac.uk',
        
        # Finance & Banking
        'paypal.com', 'stripe.com', 'square.com', 'coinbase.com', 'binance.com',
        'chase.com', 'bankofamerica.com', 'wellsfargo.com', 'capitalone.com',
        'americanexpress.com', 'discover.com', 'venmo.com', 'cashapp.com',
        
        # Travel & Hospitality
        'booking.com', 'airbnb.com', 'expedia.com', 'tripadvisor.com', 'hotels.com',
        'vrbo.com', 'kayak.com', 'skyscanner.com', 'uber.com', 'lyft.com',
        'doordash.com', 'grubhub.com', 'ubereats.com', 'postmates.com',
        
        # Communication
        'gmail.com', 'outlook.com', 'yahoo.com', 'protonmail.com', 'mail.ru',
        'whatsapp.com', 'telegram.org', 'discord.com', 'signal.org', 'messenger.com',
        
        # Gaming
        'steampowered.com', 'epicgames.com', 'roblox.com', 'minecraft.net', 'ea.com',
        'activision.com', 'ubisoft.com', 'nintendo.com', 'playstation.com', 'xbox.com',
        'leagueoflegends.com', 'fortnite.com', 'valorant.com', 'overwatchleague.com',
        
        # Software & Tools
        'adobe.com', 'canva.com', 'figma.com', 'sketch.com', 'invision.com',
        'office.com', 'docs.google.com', 'sheets.google.com', 'slides.google.com',
        'adobespark.com', 'prezi.com', 'miro.com', 'lucidchart.com',
        
        # Health & Fitness
        'webmd.com', 'mayoclinic.org', 'healthline.com', 'nih.gov', 'cdc.gov',
        'fitbit.com', 'myfitnesspal.com', 'strava.com', 'peloton.com', 'noom.com',
        
        # Government & Organizations
        'usa.gov', 'gov.uk', 'europa.eu', 'un.org', 'who.int',
        'imf.org', 'worldbank.org', 'redcross.org', 'unicef.org',
        
        # Regional/International
        'baidu.com', 'qq.com', 'taobao.com', 'tmall.com', 'jd.com',
        'weibo.com', 'vk.com', 'yandex.ru', 'naver.com', 'line.me',
        'rakuten.co.jp', 'yahoo.co.jp', 'mercadolibre.com', 'olx.com',
        
        # Additional Top Sites (to reach 200+)
        'indeed.com', 'glassdoor.com', 'behance.net', 'dribbble.com', 'producthunt.com',
        'ycombinator.com', 'techcrunch.com', 'theverge.com', 'engadget.com', 'cnet.com',
        'ign.com', 'gamespot.com', 'polygon.com', 'kotaku.com', 'pcgamer.com',
        'imdb.com', 'rottentomatoes.com', 'metacritic.com', 'letterboxd.com',
        'goodreads.com', 'amazon.co.uk', 'amazon.de', 'amazon.fr', 'amazon.jp',
        'ebay.co.uk', 'ebay.de', 'ebay.fr', 'craigslist.org', 'gumtree.com',
        'yelp.com', 'zomato.com', 'opentable.com', 'seamless.com', 'justeat.com',
        'deliveroo.com', 'foodpanda.com', 'swiggy.com', 'fandango.com', 'stubhub.com',
        'ticketmaster.com', 'eventbrite.com', 'meetup.com', 'patreon.com', 'ko-fi.com',
        'gofundme.com', 'kickstarter.com', 'indiegogo.com', 'change.org', 'care2.com',
        'reverbnation.com', 'bandcamp.com', 'mixcloud.com', 'pandora.com', 'last.fm',
        'discogs.com', 'allmusic.com', 'genius.com', 'musixmatch.com', 'shazam.com',
        'weather.com', 'accuweather.com', 'weather.gov', 'wunderground.com',
        'zillow.com', 'realtor.com', 'trulia.com', 'apartments.com', 'rent.com',
        'houzz.com', 'pinterest.com', 'tumblr.com', 'quora.com', 'stackoverflow.com'
    ]
    
    # URL path components for realistic patterns
    PATH_PATTERNS = {
        'login': [
            'login', 'signin', 'auth', 'authenticate', 'account/login',
            'user/login', 'member/signin', 'portal/login', 'sso', 'oauth'
        ],
        'docs': [
            'docs', 'documentation', 'help', 'support', 'guide', 'api',
            'reference', 'manual', 'tutorial', 'learn', 'developers',
            'api-reference', 'getting-started', 'quickstart', 'faq'
        ],
        'products': [
            'products', 'shop', 'store', 'catalog', 'browse', 'search',
            'category', 'item', 'product', 'listing', 'marketplace',
            'deals', 'offers', 'sale', 'bestsellers', 'new-arrivals'
        ],
        'account': [
            'account', 'profile', 'dashboard', 'settings', 'preferences',
            'my-account', 'user/profile', 'account/settings', 'billing',
            'subscriptions', 'orders', 'history', 'favorites', 'wishlist'
        ],
        'content': [
            'blog', 'news', 'articles', 'posts', 'stories', 'updates',
            'insights', 'resources', 'library', 'archive', 'magazine',
            'press', 'media', 'announcements', 'events', 'webinars'
        ],
        'company': [
            'about', 'about-us', 'company', 'contact', 'careers', 'jobs',
            'team', 'investors', 'press', 'legal', 'terms', 'privacy',
            'security', 'compliance', 'partners', 'affiliates'
        ]
    }
    
    # Query parameter patterns
    QUERY_PATTERNS = [
        {'q': ['search_term', 'product_name', 'query', 'keyword']},
        {'id': ['12345', '67890', 'abc123', 'xyz789']},
        {'page': ['1', '2', '5', '10']},
        {'category': ['electronics', 'books', 'clothing', 'home']},
        {'sort': ['price', 'rating', 'newest', 'popular']},
        {'filter': ['instock', 'sale', 'featured']},
        {'lang': ['en', 'es', 'fr', 'de', 'ja', 'zh']},
        {'ref': ['homepage', 'email', 'social', 'ad']},
        {'utm_source': ['google', 'facebook', 'twitter', 'linkedin']},
        {'session_id': ['sess_12345', 'sess_67890']},
    ]
    
    # Product/content identifiers
    PRODUCT_IDS = [
        'B08N5WRWNW', 'B07XJ8C8F5', 'B09JQMJHXY', 'B0B1H3F5XY',
        'prod-12345', 'item-67890', 'sku-abc123', 'art-789xyz',
        'doc-guide-2024', 'tutorial-001', 'ref-api-v2', 'help-faq-10'
    ]
    
    # Nested paths for complex URLs
    NESTED_PATHS = [
        ['en', 'us', 'products', 'electronics', 'laptops'],
        ['api', 'v2', 'users', 'profile', 'update'],
        ['docs', 'latest', 'guides', 'getting-started', 'installation'],
        ['blog', '2024', '02', 'new-features-announcement'],
        ['support', 'kb', 'articles', 'troubleshooting', 'common-issues'],
        ['shop', 'category', 'mens', 'clothing', 'shirts'],
        ['learn', 'courses', 'programming', 'python', 'advanced'],
        ['resources', 'downloads', 'software', 'tools', 'utilities'],
        ['community', 'forums', 'discussions', 'general', 'thread-12345'],
        ['account', 'settings', 'security', 'two-factor-authentication']
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize augmentor with seed for reproducibility"""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_login_urls(self, n_samples: int = 2000) -> List[Dict]:
        """Generate login page URLs"""
        urls = []
        domains = random.choices(self.TOP_DOMAINS, k=n_samples)
        
        for domain in domains:
            path = random.choice(self.PATH_PATTERNS['login'])
            
            # Sometimes add query parameters
            if random.random() < 0.3:
                query = f"?redirect={urllib.parse.quote(f'/{random.choice(self.PATH_PATTERNS['account'])}')}"
            else:
                query = ''
            
            url = f"https://{domain}/{path}{query}"
            urls.append({
                'url': url,
                'type': 'benign',
                'pattern': 'login',
                'source': 'augmented'
            })
        
        return urls
    
    def generate_docs_urls(self, n_samples: int = 3000) -> List[Dict]:
        """Generate documentation URLs"""
        urls = []
        domains = random.choices(self.TOP_DOMAINS, k=n_samples)
        
        for domain in domains:
            base_path = random.choice(self.PATH_PATTERNS['docs'])
            
            # Add nested structure
            if random.random() < 0.6:
                nested = random.choice([
                    ['api', 'reference', 'v2'],
                    ['guides', 'installation'],
                    ['tutorials', 'beginners'],
                    ['faq', 'common-questions'],
                    ['changelog', 'v3.2.1']
                ])
                path = f"{base_path}/{'/'.join(nested)}"
            else:
                path = base_path
            
            # Add anchor
            if random.random() < 0.2:
                anchor = f"#{random.choice(['introduction', 'examples', 'parameters', 'returns'])}"
            else:
                anchor = ''
            
            url = f"https://{domain}/{path}{anchor}"
            urls.append({
                'url': url,
                'type': 'benign',
                'pattern': 'documentation',
                'source': 'augmented'
            })
        
        return urls
    
    def generate_product_urls(self, n_samples: int = 4000) -> List[Dict]:
        """Generate product/shop URLs"""
        urls = []
        domains = random.choices(self.TOP_DOMAINS, k=n_samples)
        
        for domain in domains:
            base_path = random.choice(self.PATH_PATTERNS['products'])
            
            # Add product ID
            product_id = random.choice(self.PRODUCT_IDS)
            path = f"{base_path}/{product_id}"
            
            # Add query parameters (common in e-commerce)
            if random.random() < 0.5:
                params = []
                if random.random() < 0.5:
                    params.append(f"ref={random.choice(['homepage', 'search', 'recommendation'])}")
                if random.random() < 0.3:
                    params.append(f"color={random.choice(['red', 'blue', 'black', 'white'])}")
                if random.random() < 0.3:
                    params.append(f"size={random.choice(['S', 'M', 'L', 'XL'])}")
                
                query = f"?{'&'.join(params)}" if params else ''
            else:
                query = ''
            
            url = f"https://{domain}/{path}{query}"
            urls.append({
                'url': url,
                'type': 'benign',
                'pattern': 'product',
                'source': 'augmented'
            })
        
        return urls
    
    def generate_query_urls(self, n_samples: int = 5000) -> List[Dict]:
        """Generate URLs with query parameters"""
        urls = []
        domains = random.choices(self.TOP_DOMAINS, k=n_samples)
        
        for domain in domains:
            base_path = random.choice([
                'search', 'results', 'browse', 'query', 'find',
                'explore', 'discover', 'filter', 'view'
            ])
            
            # Build complex query string
            n_params = random.randint(2, 5)
            params = []
            
            for _ in range(n_params):
                param_dict = random.choice(self.QUERY_PATTERNS)
                param_name = list(param_dict.keys())[0]
                param_value = random.choice(param_dict[param_name])
                params.append(f"{param_name}={urllib.parse.quote(str(param_value))}")
            
            query = '&'.join(params)
            url = f"https://{domain}/{base_path}?{query}"
            
            urls.append({
                'url': url,
                'type': 'benign',
                'pattern': 'query_parameters',
                'source': 'augmented'
            })
        
        return urls
    
    def generate_nested_urls(self, n_samples: int = 3000) -> List[Dict]:
        """Generate deeply nested path URLs"""
        urls = []
        domains = random.choices(self.TOP_DOMAINS, k=n_samples)
        
        for domain in domains:
            # Choose nested path structure
            nested_path = random.choice(self.NESTED_PATHS)
            
            # Sometimes add more depth
            if random.random() < 0.3:
                extra = [
                    random.choice(['details', 'info', 'view', 'edit']),
                    str(random.randint(1000, 9999))
                ]
                nested_path = nested_path + extra
            
            path = '/'.join(nested_path)
            url = f"https://{domain}/{path}"
            
            urls.append({
                'url': url,
                'type': 'benign',
                'pattern': 'nested_paths',
                'source': 'augmented'
            })
        
        return urls
    
    def generate_account_urls(self, n_samples: int = 2000) -> List[Dict]:
        """Generate account/profile URLs"""
        urls = []
        domains = random.choices(self.TOP_DOMAINS, k=n_samples)
        
        for domain in domains:
            base_path = random.choice(self.PATH_PATTERNS['account'])
            
            # Add user identifier sometimes
            if random.random() < 0.4:
                user_id = f"user-{random.randint(100000, 999999)}"
                path = f"users/{user_id}/{base_path}"
            else:
                path = base_path
            
            url = f"https://{domain}/{path}"
            
            urls.append({
                'url': url,
                'type': 'benign',
                'pattern': 'account',
                'source': 'augmented'
            })
        
        return urls
    
    def generate_content_urls(self, n_samples: int = 2000) -> List[Dict]:
        """Generate blog/content URLs"""
        urls = []
        domains = random.choices(self.TOP_DOMAINS, k=n_samples)
        
        for domain in domains:
            base_path = random.choice(self.PATH_PATTERNS['content'])
            
            # Add date-based structure
            if random.random() < 0.5:
                year = random.randint(2020, 2024)
                month = f"{random.randint(1, 12):02d}"
                slug = random.choice([
                    'new-features-released',
                    'security-update',
                    'product-announcement',
                    'company-milestone',
                    'technical-deep-dive',
                    'customer-story',
                    'industry-insights',
                    'how-to-guide'
                ])
                path = f"{base_path}/{year}/{month}/{slug}"
            else:
                path = base_path
            
            url = f"https://{domain}/{path}"
            
            urls.append({
                'url': url,
                'type': 'benign',
                'pattern': 'content',
                'source': 'augmented'
            })
        
        return urls
    
    def generate_all_urls(self, total_samples: int = 21000) -> pd.DataFrame:
        """
        Generate complete set of augmented URLs
        
        Distribution:
        - Login: 2000 (9.5%)
        - Docs: 3000 (14.3%)
        - Products: 4000 (19.0%)
        - Query params: 5000 (23.8%)
        - Nested paths: 3000 (14.3%)
        - Account: 2000 (9.5%)
        - Content: 2000 (9.5%)
        
        Returns:
            DataFrame with columns: url, type
        """
        print("="*80)
        print("GENERATING AUGMENTED BRAND URLS")
        print("="*80)
        print()
        
        all_urls = []
        
        # Generate each pattern type
        patterns = [
            ('Login pages', self.generate_login_urls, 2000),
            ('Documentation', self.generate_docs_urls, 3000),
            ('Product pages', self.generate_product_urls, 4000),
            ('Query parameters', self.generate_query_urls, 5000),
            ('Nested paths', self.generate_nested_urls, 3000),
            ('Account pages', self.generate_account_urls, 2000),
            ('Content pages', self.generate_content_urls, 2000),
        ]
        
        for pattern_name, generator_func, n_samples in patterns:
            print(f"Generating {pattern_name:20} : {n_samples:5} URLs")
            urls = generator_func(n_samples)
            all_urls.extend(urls)
        
        print()
        print(f"Total generated: {len(all_urls)} URLs")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_urls)
        
        # Keep only url and type columns for consistency with existing data
        df = df[['url', 'type']]
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['url'])
        final_count = len(df)
        
        if initial_count > final_count:
            print(f"Removed {initial_count - final_count} duplicate URLs")
        
        print(f"Final augmented dataset: {final_count} unique URLs")
        print()
        
        return df


def merge_datasets(original_path: str, augmented_df: pd.DataFrame, 
                   output_path: str = 'data/malicious_phish_augmented.csv') -> pd.DataFrame:
    """
    Merge original dataset with augmented brand URLs
    
    Args:
        original_path: Path to original dataset
        augmented_df: Augmented URLs DataFrame
        output_path: Path to save merged dataset
        
    Returns:
        Merged DataFrame
    """
    print("="*80)
    print("MERGING DATASETS")
    print("="*80)
    print()
    
    # Load original dataset
    print(f"Loading original dataset from {original_path}...")
    df_original = pd.read_csv(original_path)
    print(f"  Original dataset: {len(df_original)} URLs")
    
    # Check class distribution in original
    print("\nOriginal class distribution:")
    for cls, count in df_original['type'].value_counts().items():
        pct = 100 * count / len(df_original)
        print(f"  {cls:15} : {count:7} ({pct:5.2f}%)")
    
    # Merge datasets
    print(f"\nMerging with {len(augmented_df)} augmented URLs...")
    df_merged = pd.concat([df_original, augmented_df], ignore_index=True)
    
    # Remove duplicates (prioritize original labels)
    print("Removing duplicates...")
    initial_merged = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=['url'], keep='first')
    final_merged = len(df_merged)
    
    if initial_merged > final_merged:
        print(f"  Removed {initial_merged - final_merged} duplicate URLs")
    
    print(f"\nMerged dataset: {final_merged} URLs")
    
    # New class distribution
    print("\nNew class distribution:")
    for cls, count in df_merged['type'].value_counts().items():
        pct = 100 * count / len(df_merged)
        print(f"  {cls:15} : {count:7} ({pct:5.2f}%)")
    
    # Calculate increase in benign class
    original_benign = len(df_original[df_original['type'] == 'benign'])
    new_benign = len(df_merged[df_merged['type'] == 'benign'])
    benign_increase = new_benign - original_benign
    
    print(f"\nBenign class increase: +{benign_increase} URLs ({100 * benign_increase / original_benign:.1f}%)")
    
    # Save merged dataset
    print(f"\nSaving merged dataset to {output_path}...")
    df_merged.to_csv(output_path, index=False)
    print("✓ Merged dataset saved")
    print()
    
    return df_merged


if __name__ == "__main__":
    """Generate augmented dataset"""
    
    # Initialize augmentor
    augmentor = BrandURLAugmentor(seed=42)
    
    # Generate 21,000 brand URLs
    df_augmented = augmentor.generate_all_urls(total_samples=21000)
    
    # Merge with original dataset
    df_merged = merge_datasets(
        original_path='data/malicious_phish.csv',
        augmented_df=df_augmented,
        output_path='data/malicious_phish_augmented.csv'
    )
    
    print("="*80)
    print("DATA AUGMENTATION COMPLETE")
    print("="*80)
    print()
    print(f"✓ Generated {len(df_augmented)} new benign URLs")
    print(f"✓ Merged dataset: {len(df_merged)} total URLs")
    print(f"✓ Saved to: data/malicious_phish_augmented.csv")
    print()
    print("Next steps:")
    print("  1. Retrain model with augmented data")
    print("  2. Re-run brand bias test")
    print("  3. Compare false positive rates")
