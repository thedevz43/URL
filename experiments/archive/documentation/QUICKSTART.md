# Quick Start Guide

## Step 1: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If you get an execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Step 2: Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3: Verify Installation

```powershell
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import pandas as pd; print('Pandas:', pd.__version__)"
```

## Step 4: Train Model (First Time)

```powershell
python main.py --train
```

This will:
- Load and preprocess ~651K URLs
- Train character-level CNN
- Save model to `models/url_detector.h5`
- Generate evaluation metrics
- Create visualizations

**Expected time**: 30-60 minutes on CPU, 5-10 minutes on GPU

## Step 5: Test Model

```powershell
# Test on sample URLs
python main.py --demo

# Interactive testing
python main.py --interactive

# Predict single URL
python main.py --predict "https://suspicious-site.com"
```

## Project Commands Reference

| Command | Description |
|---------|-------------|
| `python main.py --train` | Train a new model |
| `python main.py --train --epochs 30` | Train with 30 epochs |
| `python main.py --demo` | Run inference demo |
| `python main.py --interactive` | Interactive URL testing |
| `python main.py --predict <url>` | Classify single URL |
| `python main.py` | Show help |

## Troubleshooting

### Issue: TensorFlow won't install
**Solution**: Make sure you have Python 3.8-3.11 (not 3.12)

### Issue: Out of memory during training
**Solution**: Reduce batch size
```powershell
python main.py --train --batch-size 64
```

### Issue: Model file not found
**Solution**: Train the model first
```powershell
python main.py --train
```

## Directory Structure After Training

```
DNN/
├── data/
│   └── malicious_phish.csv
├── models/
│   ├── url_detector.h5          ← Trained model
│   ├── preprocessor.pkl          ← Preprocessing config
│   ├── training_history.png      ← Training curves
│   ├── evaluation_results.png    ← Performance metrics
│   └── training_metadata.json    ← Training stats
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── main.py
├── requirements.txt
└── README.md
```

## Expected Results

After training, you should see:
- **Accuracy**: ~95%
- **F1-Score**: ~91% (macro average)
- Training curves in `models/training_history.png`
- Confusion matrix in `models/evaluation_results.png`

## Next Steps

1. ✅ Review `evaluation_results.png` to understand model performance
2. ✅ Test with your own URLs using `--interactive` mode
3. ✅ Read `README.md` for detailed methodology
4. ✅ Explore `src/` files to understand implementation

## Need Help?

- Check [README.md](README.md) for full documentation
- Review code comments in `src/` files
- Each module can be run standalone for testing
