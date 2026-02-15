"""
Robustness Test Cases for URL Detection Model
Tests model behavior on edge cases and extreme inputs
"""

import random
import string
from typing import List, Dict, Any
import unicodedata


class RobustnessTestGenerator:
    """Generate edge case URLs to test model robustness"""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed"""
        random.seed(seed)
    
    def generate_empty_strings(self) -> List[Dict[str, Any]]:
        """Test empty and whitespace-only inputs"""
        return [
            {'url': '', 'type': 'empty_string', 'description': 'Empty string'},
            {'url': ' ', 'type': 'whitespace', 'description': 'Single space'},
            {'url': '   ', 'type': 'whitespace', 'description': 'Multiple spaces'},
            {'url': '\n', 'type': 'newline', 'description': 'Newline character'},
            {'url': '\t', 'type': 'tab', 'description': 'Tab character'},
            {'url': '\r\n', 'type': 'carriage_return', 'description': 'CRLF'},
        ]
    
    def generate_extremely_long_urls(self) -> List[Dict[str, Any]]:
        """Test URLs exceeding normal length limits"""
        base_url = "https://example.com/"
        
        return [
            {
                'url': base_url + 'a' * 500,
                'type': 'long_path',
                'length': 500,
                'description': '500 char path'
            },
            {
                'url': base_url + 'a' * 1000,
                'type': 'long_path',
                'length': 1000,
                'description': '1000 char path'
            },
            {
                'url': base_url + 'a' * 2000,
                'type': 'long_path',
                'length': 2000,
                'description': '2000 char path (beyond model limit)'
            },
            {
                'url': base_url + 'a' * 5000,
                'type': 'extreme_length',
                'length': 5000,
                'description': '5000 char path (extreme)'
            },
            {
                'url': 'https://' + 'subdomain.' * 50 + 'example.com/',
                'type': 'long_domain',
                'length': len('subdomain.' * 50),
                'description': '50 subdomains'
            },
            {
                'url': base_url + '?' + '&'.join([f'param{i}=value{i}' for i in range(100)]),
                'type': 'many_parameters',
                'length': None,
                'description': '100 query parameters'
            }
        ]
    
    def generate_random_noise(self, n_samples: int = 20) -> List[Dict[str, Any]]:
        """Generate random character sequences"""
        noise_urls = []
        
        for i in range(n_samples):
            length = random.randint(10, 200)
            
            if i % 4 == 0:
                # Pure random ASCII
                noise = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            elif i % 4 == 1:
                # Random with special chars
                noise = ''.join(random.choices(
                    string.ascii_letters + string.digits + '!@#$%^&*()_+-=[]{}|;:,.<>?/',
                    k=length
                ))
            elif i % 4 == 2:
                # Random with URL structure
                noise = f"http://{''.join(random.choices(string.ascii_lowercase, k=10))}.com"
            else:
                # Complete gibberish
                noise = ''.join(chr(random.randint(33, 126)) for _ in range(length))
            
            noise_urls.append({
                'url': noise,
                'type': 'random_noise',
                'length': length,
                'description': f'Random noise pattern {i%4 + 1}'
            })
        
        return noise_urls
    
    def generate_unicode_urls(self) -> List[Dict[str, Any]]:
        """Test URLs with various Unicode characters"""
        return [
            {
                'url': 'https://café.com/page',
                'type': 'unicode_ascii',
                'description': 'Common accented characters'
            },
            {
                'url': 'https://münchen.de/info',
                'type': 'unicode_german',
                'description': 'German umlauts'
            },
            {
                'url': 'https://日本.jp/ページ',
                'type': 'unicode_japanese',
                'description': 'Japanese characters'
            },
            {
                'url': 'https://中国.cn/网站',
                'type': 'unicode_chinese',
                'description': 'Chinese characters'
            },
            {
                'url': 'https://한국.kr/페이지',
                'type': 'unicode_korean',
                'description': 'Korean characters'
            },
            {
                'url': 'https://россия.ru/страница',
                'type': 'unicode_russian',
                'description': 'Cyrillic characters'
            },
            {
                'url': 'https://الإمارات.ae/صفحة',
                'type': 'unicode_arabic',
                'description': 'Arabic characters'
            },
            {
                'url': 'https://ελλάδα.gr/σελίδα',
                'type': 'unicode_greek',
                'description': 'Greek characters'
            },
            {
                'url': 'https://example.com/emoji🔥💯✨',
                'type': 'unicode_emoji',
                'description': 'Emoji in path'
            },
            {
                'url': 'https://test.com/\u200b\u200c\u200d',  # Zero-width characters
                'type': 'unicode_zero_width',
                'description': 'Zero-width characters'
            },
            {
                'url': 'https://test.com/\ufeff',  # Zero-width no-break space
                'type': 'unicode_bom',
                'description': 'Byte order mark'
            }
        ]
    
    def generate_missing_protocol(self) -> List[Dict[str, Any]]:
        """Test URLs without protocol schemes"""
        return [
            {
                'url': 'example.com',
                'type': 'no_protocol',
                'description': 'Domain only'
            },
            {
                'url': 'www.example.com',
                'type': 'no_protocol_www',
                'description': 'WWW prefix without protocol'
            },
            {
                'url': 'example.com/path/to/page',
                'type': 'no_protocol_path',
                'description': 'Domain with path, no protocol'
            },
            {
                'url': '//example.com',
                'type': 'protocol_relative',
                'description': 'Protocol-relative URL'
            },
            {
                'url': 'ftp://example.com',
                'type': 'ftp_protocol',
                'description': 'FTP protocol'
            },
            {
                'url': 'file:///etc/passwd',
                'type': 'file_protocol',
                'description': 'File protocol'
            },
            {
                'url': 'javascript:alert(1)',
                'type': 'javascript_protocol',
                'description': 'JavaScript protocol'
            },
            {
                'url': 'data:text/html,<h1>Test</h1>',
                'type': 'data_protocol',
                'description': 'Data protocol'
            }
        ]
    
    def generate_malformed_urls(self) -> List[Dict[str, Any]]:
        """Test malformed and invalid URL structures"""
        return [
            {
                'url': 'http://',
                'type': 'malformed',
                'description': 'Protocol only'
            },
            {
                'url': 'http://.',
                'type': 'malformed',
                'description': 'Protocol with single dot'
            },
            {
                'url': 'http://..',
                'type': 'malformed',
                'description': 'Protocol with double dots'
            },
            {
                'url': 'http://...',
                'type': 'malformed',
                'description': 'Protocol with triple dots'
            },
            {
                'url': 'http://example',
                'type': 'malformed',
                'description': 'No TLD'
            },
            {
                'url': 'http://.com',
                'type': 'malformed',
                'description': 'TLD only'
            },
            {
                'url': 'http://example..com',
                'type': 'malformed',
                'description': 'Double dots in domain'
            },
            {
                'url': 'http://example.com:99999',
                'type': 'malformed',
                'description': 'Invalid port'
            },
            {
                'url': 'http://example.com::80',
                'type': 'malformed',
                'description': 'Double colon in port'
            },
            {
                'url': 'http://[invalid',
                'type': 'malformed',
                'description': 'Unclosed bracket'
            },
            {
                'url': 'http://user:pass:extra@example.com',
                'type': 'malformed',
                'description': 'Invalid auth format'
            },
            {
                'url': 'http://example.com/<>',
                'type': 'malformed',
                'description': 'Angle brackets in path'
            },
            {
                'url': 'http://example.com/path with spaces',
                'type': 'malformed',
                'description': 'Unencoded spaces'
            }
        ]
    
    def generate_special_cases(self) -> List[Dict[str, Any]]:
        """Test special edge cases"""
        return [
            {
                'url': 'localhost',
                'type': 'localhost',
                'description': 'Localhost'
            },
            {
                'url': 'http://localhost:8080',
                'type': 'localhost_port',
                'description': 'Localhost with port'
            },
            {
                'url': '127.0.0.1',
                'type': 'loopback_ip',
                'description': 'Loopback IP'
            },
            {
                'url': 'http://192.168.1.1',
                'type': 'private_ip',
                'description': 'Private IP range'
            },
            {
                'url': 'http://10.0.0.1',
                'type': 'private_ip',
                'description': 'Private IP 10.x'
            },
            {
                'url': 'http://[::1]',
                'type': 'ipv6_loopback',
                'description': 'IPv6 loopback'
            },
            {
                'url': 'http://[2001:db8::1]',
                'type': 'ipv6',
                'description': 'IPv6 address'
            },
            {
                'url': None,
                'type': 'none_value',
                'description': 'None value'
            },
            {
                'url': 42,
                'type': 'integer',
                'description': 'Integer instead of string'
            },
            {
                'url': ['http://example.com'],
                'type': 'list',
                'description': 'List instead of string'
            },
            {
                'url': {'url': 'http://example.com'},
                'type': 'dict',
                'description': 'Dict instead of string'
            }
        ]
    
    def generate_encoding_attacks(self) -> List[Dict[str, Any]]:
        """Test URL encoding edge cases"""
        return [
            {
                'url': 'http://example.com/%2e%2e%2f%2e%2e%2f',
                'type': 'double_encoding',
                'description': 'Directory traversal encoded'
            },
            {
                'url': 'http://example.com/%00',
                'type': 'null_byte',
                'description': 'Null byte injection'
            },
            {
                'url': 'http://example.com/%0d%0a',
                'type': 'crlf_injection',
                'description': 'CRLF injection encoded'
            },
            {
                'url': 'http://example.com/%ff%fe',
                'type': 'invalid_encoding',
                'description': 'Invalid UTF-8 sequences'
            },
            {
                'url': 'http://example.com/%u0041',
                'type': 'unicode_escape',
                'description': 'Unicode escape sequence'
            }
        ]
    
    def generate_sql_injection_attempts(self) -> List[Dict[str, Any]]:
        """Test SQL injection patterns in URLs (should still classify safely)"""
        return [
            {
                'url': "http://example.com/?id=1' OR '1'='1",
                'type': 'sql_injection',
                'description': 'Classic SQLi'
            },
            {
                'url': "http://example.com/?id=1; DROP TABLE users--",
                'type': 'sql_injection',
                'description': 'DROP TABLE attempt'
            },
            {
                'url': "http://example.com/?id=1' UNION SELECT * FROM users--",
                'type': 'sql_injection',
                'description': 'UNION SELECT'
            }
        ]
    
    def generate_xss_attempts(self) -> List[Dict[str, Any]]:
        """Test XSS patterns in URLs"""
        return [
            {
                'url': 'http://example.com/?q=<script>alert(1)</script>',
                'type': 'xss',
                'description': 'Script tag injection'
            },
            {
                'url': 'http://example.com/?q=javascript:alert(1)',
                'type': 'xss',
                'description': 'JavaScript protocol'
            },
            {
                'url': 'http://example.com/?q=<img src=x onerror=alert(1)>',
                'type': 'xss',
                'description': 'Image onerror'
            }
        ]
    
    def generate_complete_robustness_suite(self) -> List[Dict[str, Any]]:
        """Generate comprehensive robustness test suite"""
        all_tests = []
        
        all_tests.extend(self.generate_empty_strings())
        all_tests.extend(self.generate_extremely_long_urls())
        all_tests.extend(self.generate_random_noise(20))
        all_tests.extend(self.generate_unicode_urls())
        all_tests.extend(self.generate_missing_protocol())
        all_tests.extend(self.generate_malformed_urls())
        all_tests.extend(self.generate_special_cases())
        all_tests.extend(self.generate_encoding_attacks())
        all_tests.extend(self.generate_sql_injection_attempts())
        all_tests.extend(self.generate_xss_attempts())
        
        return all_tests


def test_model_robustness(model, preprocessor, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Test model on robustness cases and capture any crashes/errors
    
    Returns:
        dict: test_results with success rate and error details
    """
    results = {
        'total_tests': len(test_cases),
        'successful': 0,
        'errors': 0,
        'crashes': [],
        'predictions': []
    }
    
    for i, test_case in enumerate(test_cases):
        try:
            url = test_case['url']
            
            # Handle non-string inputs
            if not isinstance(url, str):
                results['crashes'].append({
                    'test_id': i,
                    'input': str(url),
                    'type': test_case['type'],
                    'error': 'Non-string input',
                    'error_type': 'TypeError'
                })
                results['errors'] += 1
                continue
            
            # Try to preprocess
            url_seq = preprocessor.encode_urls([url])
            domain_seq = preprocessor.encode_domains([url])
            
            # Try to predict
            prediction = model.predict([url_seq, domain_seq], verbose=0)
            
            results['successful'] += 1
            results['predictions'].append({
                'test_id': i,
                'url': url[:100] if len(url) > 100 else url,
                'type': test_case['type'],
                'prediction': prediction[0].tolist(),
                'predicted_class': int(prediction[0].argmax())
            })
            
        except Exception as e:
            results['crashes'].append({
                'test_id': i,
                'input': str(test_case['url'])[:100] if isinstance(test_case['url'], str) else str(test_case['url']),
                'type': test_case['type'],
                'error': str(e),
                'error_type': type(e).__name__
            })
            results['errors'] += 1
    
    results['success_rate'] = results['successful'] / results['total_tests'] if results['total_tests'] > 0 else 0
    
    return results


if __name__ == '__main__':
    # Demo usage
    print("="*80)
    print("ROBUSTNESS TEST GENERATOR DEMO")
    print("="*80)
    
    gen = RobustnessTestGenerator(seed=42)
    
    print("\n1. Empty Strings:")
    for item in gen.generate_empty_strings()[:3]:
        print(f"   {item}")
    
    print("\n2. Extremely Long URLs:")
    for item in gen.generate_extremely_long_urls()[:3]:
        print(f"   Length: {item['length']}, Desc: {item['description']}")
    
    print("\n3. Unicode URLs:")
    for item in gen.generate_unicode_urls()[:3]:
        print(f"   {item['url']} ({item['description']})")
    
    print("\n4. Malformed URLs:")
    for item in gen.generate_malformed_urls()[:5]:
        print(f"   {item['url']} - {item['description']}")
    
    print(f"\n\nTotal test cases in complete suite:")
    complete_suite = gen.generate_complete_robustness_suite()
    print(f"Total: {len(complete_suite)} test cases")
    
    # Count by type
    type_counts = {}
    for case in complete_suite:
        t = case['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\nBreakdown by type:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t:25} : {count:3} tests")
