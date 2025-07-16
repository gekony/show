# ベースとなる公式Pythonイメージを選択
FROM python:3.11-slim

# 不要なキャッシュファイルを作成しないように設定
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

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

# 先にrequirements.txtだけをコピーしてライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 残りの全ファイルをコピーする
COPY . .

# 【最終診断】コピー後のファイル構造をすべて表示する
RUN ls -R

# Botを実行するコマンド
CMD ["python", "bot.py"]
