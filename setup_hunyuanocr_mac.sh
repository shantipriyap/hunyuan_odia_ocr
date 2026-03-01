#!/bin/bash
# Automated setup for HunyuanOCR training on macOS

# 1. Install Python 3.12 via Homebrew if not present
if ! command -v python3.12 &> /dev/null; then
    echo "Installing Python 3.12 via Homebrew..."
    brew install python@3.12
fi

# 2. Create and activate virtual environment
PYENV=./hunyuan_odia_ocr_env
python3.12 -m venv $PYENV
source $PYENV/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install required packages
pip install pillow peft datasets torch
pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4

# 5. Run the training script
python hunyuan_odia_ocr_train.py
