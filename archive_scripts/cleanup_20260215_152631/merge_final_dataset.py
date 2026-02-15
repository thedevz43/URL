"""
Merge International Dataset

Combines:
1. malicious_phish_augmented.csv (660,691 URLs with brand augmentation)
2. international_augmentation.csv (15,700 international URLs)

Creates final training dataset with comprehensive coverage.
"""

import pandas as pd
import numpy as np

print("="*80)
print("MERGING DATASETS FOR FINAL TRAINING")
print("="*80)
print()

# Load existing augmented dataset
print("[1/3] Loading augmented dataset...")
df_augmented = pd.read_csv('data/malicious_phish_augmented.csv')
print(f"      Loaded: {len(df_augmented):,} URLs")
print(f"      Columns: {list(df_augmented.columns)}")
print()
print("      Distribution:")
for label, count in df_augmented['type'].value_counts(). items():
    pct = 100 * count / len(df_augmented)
    print(f"        {label:15} : {count:7,} ({pct:5.2f}%)")

# Load international dataset
print()
print("[2/3] Loading international dataset...")
df_international = pd.read_csv('data/international_augmentation.csv')
print(f"      Loaded: {len(df_international):,} URLs")
print(f"      Columns: {list(df_international.columns)}")
print()
print("      Distribution:")
for label, count in df_international['type'].value_counts().items():
    pct = 100 * count / len(df_international)
    print(f"        {label:15} : {count:7,} ({pct:5.2f}%)")

# Combine datasets
print()
print("[3/3] Merging and deduplicating...")
df_combined = pd.concat([df_augmented, df_international], ignore_index=True)
print(f"      Combined size: {len(df_combined):,} URLs")

# Remove duplicates
initial_count = len(df_combined)
df_combined = df_combined.drop_duplicates(subset=['url'])
final_count = len(df_combined)
duplicates_removed = initial_count - final_count

if duplicates_removed > 0:
    print(f"      Removed {duplicates_removed:,} duplicate URLs")

print(f"      Final size: {final_count:,} URLs")

# Save final dataset
output_path = 'data/malicious_phish_final.csv'
df_combined.to_csv(output_path, index=False)

print()
print("="*80)
print("FINAL DATASET CREATED")
print("="*80)
print(f"File: {output_path}")
print(f"Total URLs: {final_count:,}")
print()
print("Final Distribution:")
for label, count in df_combined['type'].value_counts().items():
    pct = 100 * count / final_count
    print(f"  {label:15} : {count:7,} ({pct:5.2f}%)")

print()
print("Growth Summary:")
print(f"  Original dataset:     651,191 URLs")
print(f"  + Brand augmentation: 660,691 URLs (+9,500)")
print(f"  + International:      {final_count:,} URLs (+{final_count - 660691:,})")
print()

# Calculate label encodings for reference
label_map = {'benign': 0, 'defacement': 1, 'phishing': 2, 'malware': 3}
print("Label Encoding Reference:")
for label, code in label_map.items():
    count = (df_combined['type'] == label).sum()
    print(f"  {code}: {label:15} ({count:,} samples)")

print()
print("✓ Dataset merge complete!")
print("="*80)
