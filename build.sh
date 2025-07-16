#!/usr/bin/env bash
# exit on error
set -o errexit

# Pythonパッケージのビルドに必要なシステムライブラリをインストール
apt-get update && apt-get install -y build-essential

# Pythonライブラリをインストール
pip install -r requirements.txt
