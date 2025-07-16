# ベースとなる公式Pythonイメージを選択
FROM python:3.11-slim

# システムを最新の状態にし、必要なライブラリをすべてインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-jpn \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# Pythonライブラリのリストをコピーし、インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# templatesフォルダを明示的にコピーする (最重要修正箇所)
COPY templates ./templates/

# bot.pyをコピーする
COPY bot.py .

# Botを実行するコマンド
CMD ["python", "bot.py"]
