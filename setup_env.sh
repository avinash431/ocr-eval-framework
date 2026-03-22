#!/usr/bin/env bash
set -e

echo "============================================"
echo "  OCR Evaluation Framework — Setup"
echo "============================================"

# --- Detect OS ---
OS=$(uname -s)
echo "Detected OS: $OS"

# --- Python version check ---
PYTHON=""
for cmd in python3.11 python3.12 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "❌ Python 3.10+ required. Please install Python first."
    exit 1
fi
PY_VERSION=$($PYTHON --version 2>&1)
echo "Using: $PY_VERSION"

# --- Create virtual environment ---
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi
source venv/bin/activate
echo "✅ Virtual environment activated"

# --- Upgrade pip ---
pip install --upgrade pip wheel setuptools --quiet

# --- Install core dependencies ---
echo "Installing core dependencies..."
pip install pyyaml pillow tqdm requests pandas jinja2 --quiet
pip install python-Levenshtein nltk rapidfuzz --quiet
pip install openpyxl matplotlib --quiet
echo "✅ Core dependencies installed"

# --- Install Tesseract (system-level) ---
echo ""
echo "Checking Tesseract..."
if command -v tesseract &>/dev/null; then
    TESS_VER=$(tesseract --version 2>&1 | head -1)
    echo "✅ Tesseract found: $TESS_VER"
else
    echo "⚠ Tesseract not found. Install it:"
    if [ "$OS" = "Darwin" ]; then
        echo "  brew install tesseract tesseract-lang"
    else
        echo "  sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel tesseract-ocr-tam tesseract-ocr-ben"
    fi
fi
pip install pytesseract --quiet

# --- Install PaddleOCR ---
echo ""
echo "Installing PaddleOCR..."
if [ "$OS" = "Darwin" ]; then
    pip install paddlepaddle paddleocr --quiet 2>/dev/null || echo "⚠ PaddleOCR install failed — try manually: pip install paddlepaddle paddleocr"
else
    pip install paddlepaddle paddleocr --quiet 2>/dev/null || echo "⚠ PaddleOCR install failed — try: pip install paddlepaddle-gpu paddleocr (for CUDA)"
fi

# --- Install Docling ---
echo "Installing Docling..."
pip install docling --quiet 2>/dev/null || echo "⚠ Docling install failed — try manually: pip install docling"

# --- Install Surya ---
echo "Installing Surya OCR..."
pip install surya-ocr --quiet 2>/dev/null || echo "⚠ Surya install failed — try manually: pip install surya-ocr"

# --- Install Cloud SDKs ---
echo ""
echo "Installing cloud API SDKs..."
pip install azure-ai-documentintelligence --quiet 2>/dev/null || true
pip install google-cloud-vision --quiet 2>/dev/null || true
pip install boto3 --quiet 2>/dev/null || true
pip install mistralai --quiet 2>/dev/null || true
echo "✅ Cloud SDKs installed"

# --- Install PyTorch (for VLM models) ---
echo ""
echo "Checking PyTorch..."
TORCH_INSTALLED=$(python -c "import torch; print('yes')" 2>/dev/null || echo "no")
if [ "$TORCH_INSTALLED" = "no" ]; then
    echo "Installing PyTorch..."
    if [ "$OS" = "Darwin" ]; then
        pip install torch torchvision --quiet
    else
        # Check for CUDA
        if command -v nvidia-smi &>/dev/null; then
            echo "  CUDA detected — installing GPU version"
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
        else
            pip install torch torchvision --quiet
        fi
    fi
fi
pip install transformers accelerate --quiet 2>/dev/null || true
echo "✅ PyTorch + Transformers installed"

# --- Detect GPU ---
echo ""
echo "Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ Apple Silicon MPS available')
    import os, subprocess
    mem = subprocess.check_output(['sysctl', '-n', 'hw.memsize']).decode().strip()
    print(f'   Unified Memory: {int(mem) / 1e9:.0f} GB')
else:
    print('⚠ No GPU detected — CPU only')
    print('  Cloud API models will work fine.')
    print('  Open-source VLM models will be slow or may not fit in memory.')
"

# --- Copy config template ---
if [ ! -f "configs/config.local.yaml" ]; then
    cp configs/config.yaml configs/config.local.yaml
    echo ""
    echo "📝 Created configs/config.local.yaml — edit this with your API keys"
fi

echo ""
echo "============================================"
echo "  ✅ Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. Edit configs/config.local.yaml with your API keys"
echo "  3. python download_dataset.py --output-dir ./test-dataset"
echo "  4. python run_single.py --model tesseract --input <image>"
echo ""
