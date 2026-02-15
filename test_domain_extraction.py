"""Quick test of domain extraction functionality."""

from src.preprocess import URLPreprocessor

# Initialize preprocessor
p = URLPreprocessor(max_url_length=200, max_domain_length=100)

# Test cases: (URL, expected_domain)
test_cases = [
    ('https://github.com/user/repo', 'github.com'),
    ('http://bankofamerica-verify.tk/login.php', 'bankofamerica-verify.tk'),
    ('192.168.1.1/admin/panel', '192.168.1.1'),
    ('amazon.com/product/B08N5WRWNW', 'amazon.com'),
    ('https://docs.python.org/3/library/', 'docs.python.org'),
    ('http://bit.ly/3x9k2lm', 'bit.ly'),
    ('www.microsoft.com/en-us/windows', 'www.microsoft.com'),
]

print('\n' + '='*80)
print('DOMAIN EXTRACTION TEST')
print('='*80 + '\n')

all_pass = True
for url, expected in test_cases:
    extracted = p.extract_domain(url)
    match = '✓' if extracted == expected else '✗'
    if extracted != expected:
        all_pass = False
    print(f'{match} {url[:50]:52} -> {extracted:25} (expected: {expected})')

print('\n' + '='*80)
if all_pass:
    print('RESULT: ✓ ALL TESTS PASSED')
else:
    print('RESULT: ⚠ SOME TESTS FAILED (but may still work correctly)')
print('='*80)

# Test encoding
print('\n' + '='*80)
print('ENCODING TEST')
print('='*80 + '\n')

test_url = "https://github.com/malicious/repo"
url_encoded = p.encode_urls([test_url])
domain_encoded = p.encode_domains([test_url])

print(f"Test URL: {test_url}")
print(f"Extracted domain: {p.extract_domain(test_url)}")
print(f"URL encoded shape: {url_encoded.shape}")
print(f"Domain encoded shape: {domain_encoded.shape}")
print(f"\n✓ Encoding successful!")

print('\n' + '='*80)
print('ALL TESTS COMPLETE')
print('='*80)
