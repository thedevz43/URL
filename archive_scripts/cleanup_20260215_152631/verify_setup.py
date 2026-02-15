"""
Setup verification script.
Run this after installing dependencies to verify everything is configured correctly.
"""

import sys
import os

print("=" * 80)
print("SETUP VERIFICATION")
print("=" * 80)

# Check Python version
print(f"\n1. Python Version:")
print(f"   {sys.version}")
required_version = (3, 8)
current_version = sys.version_info[:2]
if current_version >= required_version and current_version < (3, 12):
    print("   ✓ Compatible Python version")
else:
    print(f"   ⚠️  Warning: Python 3.8-3.11 recommended, you have {current_version[0]}.{current_version[1]}")

# Check required packages
print(f"\n2. Required Packages:")

packages = {
    'tensorflow': '2.15.0',
    'pandas': '2.1.4',
    'numpy': '1.26.2',
    'matplotlib': '3.8.2',
    'seaborn': '0.13.0',
    'sklearn': '1.3.2'
}

all_installed = True
for package, expected_version in packages.items():
    try:
        if package == 'sklearn':
            import sklearn
            version = sklearn.__version__
            package_name = 'scikit-learn'
        else:
            module = __import__(package)
            version = module.__version__
            package_name = package
        
        print(f"   ✓ {package_name}: {version}")
    except ImportError:
        print(f"   ✗ {package_name}: NOT INSTALLED")
        all_installed = False
    except AttributeError:
        print(f"   ? {package_name}: Installed (version unknown)")

# Check dataset
print(f"\n3. Dataset:")
data_path = "data/malicious_phish.csv"
if os.path.exists(data_path):
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"   ✓ Dataset found: {data_path}")
    print(f"   ✓ Shape: {df.shape}")
    print(f"   ✓ Columns: {df.columns.tolist()}")
else:
    print(f"   ✗ Dataset not found: {data_path}")
    all_installed = False

# Check directory structure
print(f"\n4. Directory Structure:")
required_dirs = ['data', 'src', 'models']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"   ✓ {dir_name}/ exists")
    else:
        print(f"   ✗ {dir_name}/ missing")
        all_installed = False

# Check source files
print(f"\n5. Source Files:")
required_files = [
    'src/preprocess.py',
    'src/model.py',
    'src/train.py',
    'src/evaluate.py',
    'main.py',
    'requirements.txt',
    'README.md'
]
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} missing")
        all_installed = False

# TensorFlow GPU check
print(f"\n6. TensorFlow Configuration:")
try:
    import tensorflow as tf
    print(f"   TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✓ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"     - {gpu.name}")
    else:
        print(f"   ℹ️  No GPU detected (CPU will be used)")
except Exception as e:
    print(f"   ⚠️  Error checking TensorFlow: {e}")

# Final status
print("\n" + "=" * 80)
if all_installed:
    print("✓ SETUP COMPLETE - All checks passed!")
    print("\nNext steps:")
    print("  1. Train the model:")
    print("     python main.py --train")
    print("\n  2. Test on sample URLs:")
    print("     python main.py --demo")
else:
    print("⚠️  SETUP INCOMPLETE - Please fix the issues above")
    print("\nTo install missing packages:")
    print("  pip install -r requirements.txt")

print("=" * 80)
