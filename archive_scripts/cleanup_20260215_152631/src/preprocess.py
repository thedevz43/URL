"""
Preprocessing module for malicious URL detection.
Handles character-level encoding, padding, label encoding, and domain extraction.
"""

import numpy as np
import pandas as pd
import pickle
import string
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class URLPreprocessor:
    """
    Preprocessor for character-level URL encoding with domain extraction.
    
    This class handles:
    - Character vocabulary creation
    - URL-to-sequence conversion
    - Domain extraction and encoding
    - Padding/truncating to fixed length
    - Label encoding to categorical format
    """
    
    def __init__(self, max_url_length=200, max_domain_length=100):
        """
        Initialize preprocessor.
        
        Args:
            max_url_length (int): Maximum length of full URL character sequences
            max_domain_length (int): Maximum length of domain character sequences
        """
        self.max_url_length = max_url_length
        self.max_domain_length = max_domain_length
        
        # Define character vocabulary: printable ASCII characters
        # Include lowercase letters, digits, and common URL characters
        self.char_vocab = (
            string.ascii_lowercase + 
            string.digits + 
            string.punctuation + 
            ' '  # Space character
        )
        
        # Create character-to-integer mapping
        # Reserve 0 for padding, start from 1
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.char_vocab)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.char_vocab)}
        
        # Add padding token
        self.char_to_idx['<PAD>'] = 0
        self.idx_to_char[0] = '<PAD>'
        
        self.vocab_size = len(self.char_to_idx)
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        self.num_classes = None
        
        print(f"Initialized URLPreprocessor:")
        print(f"  - Vocabulary size: {self.vocab_size}")
        print(f"  - Max URL length: {self.max_url_length}")
        print(f"  - Max domain length: {self.max_domain_length}")
        print(f"  - Character set: {self.char_vocab[:20]}...")
    
    def extract_domain(self, url):
        """
        Extract domain from URL.
        
        Handles various URL formats:
        - http://example.com/path -> example.com
        - https://sub.example.com/path -> sub.example.com
        - example.com/path -> example.com
        - //example.com/path -> example.com
        - 192.168.1.1/admin -> 192.168.1.1
        
        Args:
            url (str): URL string
            
        Returns:
            str: Extracted domain
        """
        # Convert to lowercase
        url = url.lower().strip()
        
        # Add scheme if missing for proper parsing
        if not url.startswith(('http://', 'https://', '//')):
            url = 'http://' + url
        
        try:
            # Parse URL
            parsed = urlparse(url)
            
            # Extract domain (netloc)
            domain = parsed.netloc
            
            # If netloc is empty, try to extract from path
            if not domain:
                # Handle cases like "example.com/path" without scheme
                parts = parsed.path.split('/')
                if parts:
                    domain = parts[0]
            
            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]
            
            # If still empty, return original URL (fallback)
            if not domain:
                domain = url
            
            return domain
            
        except Exception:
            # Fallback: try simple regex extraction
            match = re.search(r'([a-z0-9\-\.]+\.[a-z]{2,}|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})', url)
            if match:
                return match.group(1)
            else:
                # Last resort: return first part before /
                return url.split('/')[0]
    
    def url_to_sequence(self, url):
        """
        Convert a single URL string to a sequence of integers.
        
        Args:
            url (str): URL string
            
        Returns:
            list: Integer sequence representing the URL
        """
        # Convert to lowercase for consistency
        url = url.lower()
        
        # Convert each character to its index
        # Unknown characters are ignored (not mapped)
        sequence = [self.char_to_idx.get(char, 0) for char in url]
        
        return sequence
    
    def encode_urls(self, urls):
        """
        Encode multiple URLs to padded sequences.
        
        Args:
            urls (list or pd.Series): Collection of URL strings
            
        Returns:
            np.ndarray: Padded integer sequences of shape (n_samples, max_url_length)
        """
        # Convert URLs to sequences
        sequences = [self.url_to_sequence(url) for url in urls]
        
        # Pad/truncate to max_url_length
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_url_length,
            padding='post',  # Pad at the end
            truncating='post'  # Truncate at the end if too long
        )
        
        return padded_sequences
    
    def encode_domains(self, urls):
        """
        Extract and encode domains from URLs.
        
        Args:
            urls (list or pd.Series): Collection of URL strings
            
        Returns:
            np.ndarray: Padded domain sequences of shape (n_samples, max_domain_length)
        """
        # Extract domains
        domains = [self.extract_domain(url) for url in urls]
        
        # Convert domains to sequences
        sequences = [self.url_to_sequence(domain) for domain in domains]
        
        # Pad/truncate to max_domain_length
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_domain_length,
            padding='post',
            truncating='post'
        )
        
        return padded_sequences
    
    def encode_labels(self, labels, fit=True):
        """
        Encode string labels to categorical format.
        
        Args:
            labels (list or pd.Series): String labels
            fit (bool): Whether to fit the label encoder (True for training data)
            
        Returns:
            tuple: (encoded_labels, one_hot_labels)
                - encoded_labels: Integer encoded labels
                - one_hot_labels: One-hot encoded labels
        """
        if fit:
            # Fit and transform for training data
            encoded = self.label_encoder.fit_transform(labels)
            self.num_classes = len(self.label_encoder.classes_)
            print(f"\nLabel encoding:")
            for idx, label in enumerate(self.label_encoder.classes_):
                print(f"  {label}: {idx}")
        else:
            # Only transform for test data
            encoded = self.label_encoder.transform(labels)
        
        # Convert to one-hot encoding
        one_hot = to_categorical(encoded, num_classes=self.num_classes)
        
        return encoded, one_hot
    
    def decode_label(self, encoded_label):
        """
        Convert integer label back to string.
        
        Args:
            encoded_label (int): Integer encoded label
            
        Returns:
            str: Original string label
        """
        return self.label_encoder.inverse_transform([encoded_label])[0]
    
    def save(self, filepath):
        """
        Save preprocessor to disk.
        
        Args:
            filepath (str): Path to save the preprocessor
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Load preprocessor from disk.
        
        Args:
            filepath (str): Path to load the preprocessor from
            
        Returns:
            URLPreprocessor: Loaded preprocessor instance
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def load_and_preprocess_data(data_path, test_size=0.2, random_state=42, multi_input=True):
    """
    Load dataset and prepare train-test split.
    
    This function:
    1. Loads the CSV file
    2. Removes duplicates and handles missing values
    3. Creates stratified train-test split
    4. Encodes URLs and labels
    5. Optionally extracts and encodes domains for multi-input model
    
    Args:
        data_path (str): Path to the CSV file
        test_size (float): Fraction of data for testing
        random_state (int): Random seed for reproducibility
        multi_input (bool): If True, returns domain encodings for multi-input model
        
    Returns:
        If multi_input=False:
            tuple: (X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, preprocessor)
        If multi_input=True:
            tuple: ([X_train_url, X_train_domain], [X_test_url, X_test_domain], 
                    y_train, y_test, y_train_cat, y_test_cat, preprocessor)
    """
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Initial dataset shape: {df.shape}")
    
    # Data validation
    print("\nData validation:")
    print(f"  - Missing values: {df.isnull().sum().sum()}")
    print(f"  - Duplicates: {df.duplicated().sum()}")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"  - Shape after removing duplicates: {df.shape}")
    
    # Remove any rows with missing URLs or labels
    df = df.dropna()
    
    # Display class distribution
    print(f"\nClass distribution:")
    class_counts = df['type'].value_counts()
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    # Extract URLs and labels
    urls = df['url'].values
    labels = df['type'].values
    
    # Initialize preprocessor
    preprocessor = URLPreprocessor(max_url_length=200, max_domain_length=100)
    
    # Stratified train-test split
    print(f"\nPerforming stratified train-test split ({int((1-test_size)*100)}/{int(test_size*100)})...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        urls,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Maintain class distribution
    )
    
    print(f"  Train size: {len(X_train_raw)}")
    print(f"  Test size: {len(X_test_raw)}")
    
    if multi_input:
        # Encode both URLs and domains
        print(f"\nEncoding URLs to character sequences...")
        X_train_url = preprocessor.encode_urls(X_train_raw)
        X_test_url = preprocessor.encode_urls(X_test_raw)
        print(f"  URL encoded shape: {X_train_url.shape}")
        
        print(f"\nExtracting and encoding domains...")
        X_train_domain = preprocessor.encode_domains(X_train_raw)
        X_test_domain = preprocessor.encode_domains(X_test_raw)
        print(f"  Domain encoded shape: {X_train_domain.shape}")
        
        # Show example
        example_url = X_train_raw[0]
        example_domain = preprocessor.extract_domain(example_url)
        print(f"\n  Example URL: '{example_url[:60]}...'")
        print(f"  Extracted domain: '{example_domain}'")
        print(f"  URL sequence: {X_train_url[0][:20]}...")
        print(f"  Domain sequence: {X_train_domain[0][:20]}...")
        
        X_train = [X_train_url, X_train_domain]
        X_test = [X_test_url, X_test_domain]
    else:
        # Encode only URLs (for single-input model)
        print(f"\nEncoding URLs to character sequences...")
        X_train = preprocessor.encode_urls(X_train_raw)
        X_test = preprocessor.encode_urls(X_test_raw)
        print(f"  Encoded shape: {X_train.shape}")
        print(f"  Example: '{X_train_raw[0][:50]}...' -> {X_train[0][:20]}...")
    
    # Encode labels
    print(f"\nEncoding labels...")
    y_train, y_train_cat = preprocessor.encode_labels(y_train_raw, fit=True)
    y_test, y_test_cat = preprocessor.encode_labels(y_test_raw, fit=False)
    
    print(f"\nPreprocessing complete!")
    if multi_input:
        print(f"  X_train_url shape: {X_train[0].shape}")
        print(f"  X_train_domain shape: {X_train[1].shape}")
    else:
        print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train_cat shape: {y_train_cat.shape}")
    print(f"  Number of classes: {preprocessor.num_classes}")
    
    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, preprocessor


if __name__ == "__main__":
    # Test the preprocessor
    data_path = "../malicious_phish.csv"
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, preprocessor = \
        load_and_preprocess_data(data_path)
    
    # Save preprocessor
    preprocessor.save("../models/preprocessor.pkl")
    
    print("\n" + "=" * 80)
    print("Preprocessing test complete!")
    print("=" * 80)
