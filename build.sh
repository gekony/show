#!/usr/bin/env bash
# exit on error
set -o errexit

# 必要なシステムライブラリとTesseract OCR本体をインストール
apt-get update
apt-get install -y \
  build-essential \
  tesseract-ocr \
  libtesseract-dev \
  libleptonica-dev \
  pkg-config

# Pythonライブラリをインストール
pip install -r requirements.txt