# Automated Setup Script for Windows PowerShell
# Run this to set up the entire environment automatically

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  MALICIOUS URL DETECTION - AUTOMATED SETUP" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python version
Write-Host "[1/5] Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  $pythonVersion" -ForegroundColor Green
    
    # Extract version number
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        
        if ($major -eq 3 -and $minor -ge 8 -and $minor -le 11) {
            Write-Host "  ✓ Python version is compatible" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ Warning: Python 3.8-3.11 recommended for TensorFlow 2.15" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "  ✗ Error: Python not found. Please install Python 3.8-3.11" -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment
Write-Host ""
Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow

if (Test-Path "venv") {
    Write-Host "  ℹ Virtual environment already exists" -ForegroundColor Cyan
} else {
    python -m venv venv
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
}

# Step 3: Activate virtual environment
Write-Host ""
Write-Host "[3/5] Activating virtual environment..." -ForegroundColor Yellow

# Check execution policy
$executionPolicy = Get-ExecutionPolicy -Scope CurrentUser
if ($executionPolicy -eq "Restricted") {
    Write-Host "  ℹ Setting execution policy to RemoteSigned..." -ForegroundColor Cyan
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
}

# Activate
& ".\venv\Scripts\Activate.ps1"
Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green

# Step 4: Install dependencies
Write-Host ""
Write-Host "[4/5] Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ All dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Some packages may have failed to install" -ForegroundColor Yellow
}

# Step 5: Verify setup
Write-Host ""
Write-Host "[5/5] Verifying setup..." -ForegroundColor Yellow

python verify_setup.py

# Final message
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Train the model (takes 30-60 minutes on CPU):" -ForegroundColor White
Write-Host "     python main.py --train" -ForegroundColor Green
Write-Host ""
Write-Host "  2. Test on sample URLs:" -ForegroundColor White
Write-Host "     python main.py --demo" -ForegroundColor Green
Write-Host ""
Write-Host "  3. Interactive mode:" -ForegroundColor White
Write-Host "     python main.py --interactive" -ForegroundColor Green
Write-Host ""
Write-Host "For help:" -ForegroundColor White
Write-Host "  python main.py" -ForegroundColor Green
Write-Host ""
Write-Host "Documentation:" -ForegroundColor White
Write-Host "  - README.md (comprehensive guide)" -ForegroundColor Cyan
Write-Host "  - QUICKSTART.md (quick reference)" -ForegroundColor Cyan
Write-Host "  - PROJECT_SUMMARY.txt (project overview)" -ForegroundColor Cyan
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
